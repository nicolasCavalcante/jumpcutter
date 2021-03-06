import os
import shutil
from pathlib import Path

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import typer
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from pytube import YouTube
from scipy.io import wavfile

TEMP_FOLER = Path(__file__).with_name("TEMP")


def main(
    input_file: str = typer.Option(
        "", help="The video file you want modified"
    ),
    url: str = typer.Option("", help="A youtube url to download and process"),
    output_file: str = typer.Option(
        "",
        help=(
            "The output file. (optional. if not included, it'll just "
            "modify the input file name)"
        ),
    ),
    silent_threshold: float = typer.Option(
        0.03,
        help=(
            "The volume amount that frames' audio needs to surpass to be"
            " consider 'sounded'. It ranges from 0 (silence) to 1 (max volume)"
        ),
    ),
    sounded_speed: float = typer.Option(
        1.00,
        help=(
            "The speed that sounded (spoken) frames "
            "should be played at. Typically 1."
        ),
    ),
    silent_speed: float = typer.Option(
        5.00,
        help=(
            "The speed that silent frames should be "
            "played at. 999999 for jumpcutting."
        ),
    ),
    frame_margin: float = typer.Option(
        1, help=("Number of silent frames adjacent to sounded frames.")
    ),
    sample_rate: float = typer.Option(
        44100, help="Sample rate of the input and output videos"
    ),
    frame_rate: float = typer.Option(
        30, help=("Frame rate of the input and output videos.")
    ),
    frame_quality: int = typer.Option(3, help=("1=highest, 31=lowest")),
    pick_tresh: bool = typer.Option(False),
):
    """
    Modifies a video file to play at different speeds when there is sound vs.
    silence.
    """
    input_file: Path = Path(
        downloadYoutubeFile(url) if url else input_file
    ).resolve()
    output_file = Path(
        output_file
        if output_file
        else input_file.with_name(
            input_file.stem + "_ALTERED" + input_file.suffix
        )
    )
    audio_fade_envelope_size = 400
    temp_folder, temp_audio = create_TEMP_FOLER(input_file)
    frame_rate = get_frame_rate(input_file, frame_rate)
    assert frame_rate > 0, "must be greater then zero"

    # extract audio
    print(f"Extracting audio file:{temp_audio}")
    ffmpeg.input(input_file.as_posix()).output(
        temp_audio.as_posix(), ab="160k", ac=2, ar=str(int(sample_rate))
    ).run(capture_stdout=True, capture_stderr=True)
    print()
    in_sample_rate, audio_data = wavfile.read(temp_audio)
    sample_rate = in_sample_rate if in_sample_rate else sample_rate

    sample_count = audio_data.shape[0]
    max_volume = abs(audio_data).max()
    samples_per_frame = sample_rate / frame_rate
    frame_count = int(np.ceil(sample_count / samples_per_frame))
    if pick_tresh:
        silent_threshold = pick_threshold(sample_rate, audio_data)
        print(f"silent_threshold = {silent_threshold}")

    chunk_has_loud_audio = flag_loud_audio_chunks(
        frame_count,
        samples_per_frame,
        sample_count,
        audio_data,
        max_volume,
        silent_threshold,
    )

    speedChangeList = compute_speed_change_list(
        frame_count,
        frame_margin,
        chunk_has_loud_audio,
        silent_speed,
        sounded_speed,
    )

    ffmpeg.input(input_file.as_posix()).output(
        (temp_folder / "frame%06d.jpg").as_posix(),
        **{"qscale:v": frame_quality},
    ).run()

    outputaudio_data = np.zeros((0, audio_data.shape[1]))
    outputPointer = 0
    lastExistingFrame = None
    premask = np.arange(audio_fade_envelope_size) / audio_fade_envelope_size
    mask = np.repeat(premask[:, np.newaxis], 2, axis=1)
    for start_stop_cnt, speedChange in enumerate(speedChangeList):
        startFrame = speedChange[0]
        stopFrame = speedChange[1]
        speed = speedChange[2]
        print(
            f" - SpeedChanges: {start_stop_cnt} of {len(speedChangeList)}",
            f" NumFrames:{stopFrame-startFrame}",
        )

        audioChunk = audio_data[
            int(startFrame * samples_per_frame) : int(
                stopFrame * samples_per_frame
            )
        ]
        alteredaudio_data, length = change_audio_speed(
            audioChunk, sample_rate, speed, temp_folder
        )
        endPointer = outputPointer + length
        outputaudio_data = np.concatenate(
            (outputaudio_data, alteredaudio_data / max_volume)
        )

        smooth_audio_transition_between_speeds(
            outputaudio_data,
            length,
            mask,
            audio_fade_envelope_size,
            outputPointer,
            endPointer,
        )
        copy_frames_output_based_on_speed(
            outputPointer,
            samples_per_frame,
            endPointer,
            startFrame,
            speed,
            lastExistingFrame,
            temp_folder,
        )
        outputPointer = endPointer

    wavfile.write(
        (temp_folder / "audioNew.wav").as_posix(),
        sample_rate,
        outputaudio_data,
    )

    input_video = ffmpeg.input(
        (temp_folder / "newFrame%06d.jpg").as_posix(), framerate=frame_rate
    )
    input_audio = ffmpeg.input((temp_folder / "audioNew.wav").as_posix())
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(
        output_file.as_posix(), strict="-2"
    ).overwrite_output().run()

    delete_path(temp_folder)


def downloadYoutubeFile(url):
    streams = YouTube(url).streams.order_by("resolution")
    audio_codec = (set([s.audio_codec for s in streams]) - {None}).pop()
    streams = streams.filter(audio_codec=audio_codec)
    name = streams.last().download()
    newname = name.replace(" ", "_")
    os.rename(name, newname)
    return newname


def create_TEMP_FOLER(input_file: Path):
    temp_folder = TEMP_FOLER / input_file.stem
    if temp_folder.exists():
        shutil.rmtree(temp_folder)
    temp_folder.mkdir(exist_ok=True, parents=True)
    temp_audio = temp_folder / "audio.wav"
    return temp_folder, temp_audio


def get_frame_rate(input_file: Path, frame_rate: float):
    vid = ffmpeg.probe(input_file.as_posix())
    streams = list(
        filter(lambda s: s["codec_type"] == "video", vid["streams"])
    )
    if streams:
        stream = streams[0]
        frame_rate = (int(stream["nb_frames"]) + 1) / float(stream["duration"])
    return frame_rate


def zoom_factory(ax, x, y, base_scale=2.0):
    def zoom_fun(event):
        cur_xlim = ax.get_xlim()
        xdata = event.xdata
        if event.button == "up":
            scale_factor = 1 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            scale_factor = 1
            print(event.button)
        newlim = np.array(cur_xlim)
        newlim = xdata + (newlim - xdata) * scale_factor

        ax.set_xlim(newlim)
        line = ax.lines[0]
        newx = np.linspace(*newlim, 1000)
        newy = np.interp(newx, x, y)
        line.set_xdata(newx)
        line.set_ydata(newy)
        plt.draw()

    ax.plot(x, y)
    ax.set_ylim(y.min(), y.max())
    fig = ax.get_figure()
    fig.canvas.mpl_connect("scroll_event", zoom_fun)

    return zoom_fun


def pick_threshold(sample_rate, audio_data):
    def onclick(event):
        print(
            "%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f"
            % (
                "double" if event.dblclick else "single",
                event.button,
                event.x,
                event.y,
                event.xdata,
                event.ydata,
            )
        )
        tresh.append(event.ydata)
        plt.close()

    tresh = []
    skiping = sample_rate // 500
    data_x = np.arange(audio_data.shape[0])[::skiping] / sample_rate
    audio_data = abs(audio_data.T[0])[::skiping]
    audio_data = audio_data / audio_data.max()
    fig, ax = plt.subplots()
    scale = 2
    zoom_factory(ax, data_x, audio_data, base_scale=scale)
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    return tresh[0]


def flag_loud_audio_chunks(
    frame_count,
    samples_per_frame,
    sample_count,
    audio_data,
    max_volume,
    silent_threshold,
):
    chunk_has_loud_audio = np.zeros((frame_count))
    for i in range(frame_count):
        start = int(i * samples_per_frame)
        end = min(int((i + 1) * samples_per_frame), sample_count)
        audiochunks = audio_data[start:end]
        maxchunksVolume = float(abs(audiochunks).max()) / max_volume
        if maxchunksVolume >= silent_threshold:
            chunk_has_loud_audio[i] = 1
    return chunk_has_loud_audio


def compute_speed_change_list(
    frame_count,
    frame_spreadage,
    chunk_has_loud_audio,
    silent_speed,
    sounded_speed,
):
    # FrameNumberStart, FrameNumberStop, speed
    chunks = [[0, 0, 0]]
    frameSpeed = np.zeros((frame_count))
    new_speeds = [silent_speed, sounded_speed]
    for i in range(frame_count):
        start = int(max(0, i - frame_spreadage))
        end = int(min(frame_count, i + 1 + frame_spreadage))
        isLoud = int(np.max(chunk_has_loud_audio[start:end]))
        frameSpeed[i] = new_speeds[isLoud]
        if i >= 1 and frameSpeed[i] != frameSpeed[i - 1]:  # Did we flip?
            chunks.append([chunks[-1][1], i, frameSpeed[i - 1]])

    chunks.append([chunks[-1][1], frame_count, frameSpeed[i - 1]])
    chunks = chunks[1:]
    return chunks


def change_audio_speed(audioChunk, sample_rate, speed, temp_folder):
    startWavFile = (temp_folder / "tempStart.wav").as_posix()
    endWavFile = (temp_folder / "tempEnd.wav").as_posix()
    wavfile.write(startWavFile, sample_rate, audioChunk)
    with WavReader(startWavFile) as reader:
        with WavWriter(
            endWavFile, reader.channels, reader.samplerate
        ) as writer:
            tsm = phasevocoder(reader.channels, speed=speed)
            tsm.run(reader, writer)
    _, alteredaudio_data = wavfile.read(endWavFile)
    length = alteredaudio_data.shape[0]
    return (alteredaudio_data, length)


def smooth_audio_transition_between_speeds(
    outputaudio_data,
    length,
    mask,
    audio_fade_envelope_size,
    outputPointer,
    endPointer,
):
    if length < audio_fade_envelope_size:
        # audio is less than 0.01 sec, let's just remove it.
        outputaudio_data[outputPointer:endPointer] = 0
    else:
        outputaudio_data[
            outputPointer : outputPointer + audio_fade_envelope_size
        ] *= mask
        outputaudio_data[
            endPointer - audio_fade_envelope_size : endPointer
        ] *= (1 - mask)


def copy_frames_output_based_on_speed(
    outputPointer,
    samples_per_frame,
    endPointer,
    startFrame,
    speed,
    lastExistingFrame,
    temp_folder,
):
    startOutputFrame = int(np.ceil(outputPointer / samples_per_frame))
    endOutputFrame = int(np.ceil(endPointer / samples_per_frame))
    for outputFrame in range(startOutputFrame, endOutputFrame):
        inputFrame = int(startFrame + speed * (outputFrame - startOutputFrame))
        didItWork = copy_frame(inputFrame, outputFrame, temp_folder)
        if didItWork:
            lastExistingFrame = inputFrame
        else:
            copy_frame(lastExistingFrame, outputFrame, temp_folder)


def copy_frame(inputFrame, outputFrame, temp_folder):
    src = (temp_folder / f"frame{inputFrame+1:06d}.jpg").as_posix()
    dst = (temp_folder / f"newFrame{outputFrame+1:06d}.jpg").as_posix()
    if not os.path.isfile(src):
        return False
    shutil.copyfile(src, dst)
    return True


def delete_path(s):
    try:
        shutil.rmtree(s, ignore_errors=False)
    except OSError:
        print("Deletion of the directory %s failed" % s)
        print(OSError)


def app():
    typer.run(main)


if __name__ == "__main__":
    app()
