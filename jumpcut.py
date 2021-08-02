import dataclasses as dc
import subprocess
import tempfile as tf
import argparse
import fractions
import pathlib
import itertools
import json
import os

import numpy as np
from tqdm import tqdm
from audiotsm import phasevocoder
from audiotsm.io.array import ArrayReader, ArrayWriter
from scipy.io import wavfile


from typing import Sequence, TypeVar, Iterable, Iterator
T = TypeVar("T", float, int)


def sh(cmd: str, save=False) -> str:
    return subprocess.run(cmd, shell=True, capture_output=save, text=save).stdout


@dc.dataclass(frozen=True)
class Chunk:
    start_frame: int
    size: int
    end_frame: int
    is_sounded: bool

    # It'd make sense for Chunk to bear this np.ndarray view from its,
    # initialization, but that leads to more complex code.
    def get_audio_segment(self, audio_samples: np.ndarray, samples_per_frame: float) -> np.ndarray:
        """Returns a segment of the total audio samples which corresponds to
        the audio belonging to the set of frames described by this chunk."""
        fst_sample = round(self.start_frame * samples_per_frame)
        lst_sample = round(self.end_frame * samples_per_frame)
        return audio_samples[fst_sample: lst_sample]


# https://stackoverflow.com/questions/18275023/
class CustomHelpFormatter(argparse.HelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ', '.join(action.option_strings) + ' ' + args_string


def phase_vocode(audio_data: np.ndarray, speed: float) -> np.ndarray:
    """Applies phase vocoding to a 'np.ndarray' representing WAV data."""
    reader = ArrayReader(audio_data.transpose())
    writer = ArrayWriter(reader.channels)
    phasevocoder(reader.channels, speed=speed).run(reader, writer)
    return writer.data.transpose()


def get_max_amplitude(s: "np.ndarray[T]") -> "T":
    return max(s.max(), -s.min())


def iter_len(it: Iterable) -> int:
    counter = 0
    for item in it:
        counter += 1
    return counter


def iter_with_margin(seq: Sequence, margin: int) -> Iterator[Sequence]:
    """ Yields slices of a sequence, one for each member, with a padding made
    of adjascent elements from both sides, totalling 'n' elements in each
    slice when possible. Reminds me of kernels in convolutional NNs.

    iter_id_kernel("ABCDEFGHIJ", 2) -> ABC ABCD ABCDE BCDEF ... FGHIJ GHIJ HIJ
    iter_id_kernel("ABCDEFGHIJ", 1) -> AB ABC BCD CDE ... GHI HIJ IJ
    """

    margin += 1

    for i in range(margin):
        yield seq[0: i + margin]

    for i in range(margin + 1, len(seq) - margin + 2):
        yield seq[i - margin: i + margin - 1]

    for i in range(len(seq) - margin + 2, len(seq) + 1):
        yield seq[i - margin: len(seq)]


def chunkify(samples_by_frame: list[np.ndarray], threshold: float, margin: int) -> Iterator[Chunk]:
    """Groups samples, corresponding to same-soundedness frames and a margin
    surrounding sounded frames, into 'Chunk' objects."""

    is_frame_sounded_single = [get_max_amplitude(chunk) > threshold 
            for chunk in samples_by_frame]

    is_frame_sounded_margin = (any(chunk)
            for chunk in iter_with_margin(is_frame_sounded_single, margin))

    total_len = 0
    for key, group in itertools.groupby(is_frame_sounded_margin):
        yield Chunk(total_len, (size := iter_len(group)), (total_len := total_len + size), key)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter,
        description="Make a video play at a different speed when it's silent.")

    parser.add_argument('src', type=str, help="path to the source/input video file.")
    parser.add_argument('dst', type=str, help="Path to the destination/output file.")

    parser.add_argument('-s', '--speed', type=float, default=5.00,
        help="the speed that silent frames should be played at")
    parser.add_argument('-t', '--threshold', type=float, default=0.03,
        help="how loud, from 0 (0%%) to 1 (100%%), a frame needs to be to not be considered silent")
    parser.add_argument('-m', '--margin', type=int, default=1,
        help="how many silent frames to keep adjascent to a non-silent video segment, for context")
    parser.add_argument('-q', '--quality', type=int, default=3,
        help="quality of frames to be extracted from input video (1=highest, 31=lowest, 3=default)")
    parser.add_argument('-d', '--directory', default=None,
        help="extract video data to this folder instead of doing it to a temporary one")
    parser.add_argument('-f', '--force', action="store_true",
        help="do not re-use the previously extracted files present in '--directory'.")

    return parser.parse_args()


def jumpcut(src: str, dst: str, speed: float, threshold: float, margin: int, quality: int, root_dir: str, force: bool) -> None:

    # This way, 'frame_count' is ensured to be integer and correct.
    cmd = f"ffprobe -v error -select_streams v:0 -count_packets -of json -show_entries stream=nb_read_packets,r_frame_rate '{src}'"
    cmd_data = json.loads(sh(cmd, save=True))
    frame_rate = float(fractions.Fraction(cmd_data["streams"][0]["r_frame_rate"]))
    frame_count = int(cmd_data["streams"][0]["nb_read_packets"])

    (all_frames_dir := root_dir / "all_frames").mkdir(exist_ok=True)
    (new_frames_dir := root_dir / "new_frames").mkdir(exist_ok=True)

    frame_fmt = "%06d.jpg"
    old_audio_path = root_dir / "old-audio.wav"
    new_audio_path = root_dir / "new-audio.wav"

    if next(all_frames_dir.iterdir(), None) is None or force:
        sh(f"ffmpeg -hide_banner -y -i '{src}' -qscale:v {quality} '{all_frames_dir / frame_fmt}'")

    if not old_audio_path.is_file() or force:
        sh(f"ffmpeg -hide_banner -y -i '{src}' -ab 160k -ac 2 -vn '{old_audio_path}'")

    sample_rate, raw_audio = wavfile.read(old_audio_path)
    samples_per_frame = sample_rate / frame_rate
    sample_count = raw_audio.shape[0]
    max_amplitude = get_max_amplitude(raw_audio)

    audio_samples_by_frame = np.array_split(raw_audio, frame_count)

    audio_out = np.empty_like(raw_audio)
    curr_sample = 0

    lastExistingFrame = None
    for chunk in tqdm(chunkify(audio_samples_by_frame, threshold * max_amplitude, margin)):

        audio_chunk = chunk.get_audio_segment(raw_audio, samples_per_frame)

        final_audio_chunk = audio_chunk if chunk.is_sounded else phase_vocode(audio_chunk, speed)
        final_audio_chunk_len = final_audio_chunk.shape[0]

        chunk_end_sample = curr_sample + final_audio_chunk_len
        audio_out[curr_sample: chunk_end_sample] = final_audio_chunk

        # First and last frames corresponding to the original chunk of audio corresponding to this one
        chunk_1st_frame = round(curr_sample / samples_per_frame)
        chunk_end_frame = round(chunk_end_sample / samples_per_frame)

        for frame in range(chunk_1st_frame, chunk_end_frame):
            inputFrame = chunk.start_frame + int((1 if chunk.is_sounded else speed) * (frame - chunk_1st_frame))
            # Marks frames to be used in the later video
            frame_src = all_frames_dir / f"{inputFrame + 1:06d}.jpg"
            frame_dst = new_frames_dir / f"{frame + 1:06d}.jpg"
            os.rename(frame_src, frame_dst)

        curr_sample = chunk_end_sample

    wavfile.write(new_audio_path, sample_rate, audio_out)

    sh(f"ffmpeg -hide_banner -y -framerate {frame_rate} -i '{new_frames_dir / frame_fmt}' -i '{new_audio_path}' -strict -2 '{dst}'")


def main(directory: str=None, **kwargs) -> None:

    if directory is None:
        with tf.TemporaryDirectory() as dir_str:
            jumpcut(**kwargs, root_dir=pathlib.Path(dir_str))
    else:
        root_dir = pathlib.Path(directory)
        root_dir.mkdir(parents=True, exist_ok=True)
        jumpcut(**kwargs, root_dir=root_dir)


if __name__ == "__main__":
    main(**vars(get_args()))

