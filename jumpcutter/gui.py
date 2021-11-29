import inspect
import sys
from multiprocessing import Process, Queue
from typing import IO

import PySimpleGUI as sg

import jumpcutter.cli as cli


class RedirectProcess(IO):
    def __init__(self, queue: Queue):
        self.queue = queue

    def write(self, string):
        self.queue.put(string)


def gui():
    # sg.theme("Dark Grey 11")
    queue = Queue()
    process: Process = None
    spec = inspect.getfullargspec(cli.main)
    layout = []
    for arg, default in zip(spec.args, spec.defaults):
        lay = FText(arg, default.default, default.help)
        if arg == "input_file":
            lay.append(sg.FileBrowse())
        elif arg == "output_file":
            lay.append(sg.FileSaveAs())
        elif arg == "pick_tresh":
            lay[1] = sg.Checkbox(
                "Pick treshold",
                default=default.default,
                key=arg + "_input",
                tooltip=default.help,
            )
        layout.append(lay)
    layout.append([sg.Text("Constructed Command Line:")])
    layout.append(
        [
            sg.Text(
                size=(80, 3),
                key="command_line",
                text_color="yellow",
                font="Courier 8",
            )
        ]
    )
    layout.append(
        [
            sg.MLine(
                size=(80, 10),
                reroute_stdout=True,
                reroute_stderr=True,
                reroute_cprint=True,
                write_only=True,
                font="Courier 8",
                autoscroll=True,
                key="ml_line",
            )
        ]
    )
    layout.append([sg.Button("Start"), sg.Button("Exit")])
    window = sg.Window("Jump Cutter", layout, finalize=True)
    while True:
        event, values = window.read(100)
        while not queue.empty():
            element = queue.get()
            if element == "Done":
                window["Start"].update(
                    "Start", button_color=sg.theme_button_color()
                )
                print("Process Done!")
                window.refresh()
            else:
                if len(element) > 1:
                    print(element + "\n")
        if event == sg.WIN_CLOSED or event == "Exit":
            break
        elif event == "Start":
            if window["Start"].ButtonText == "Stop":
                process.kill()
                process.join()
                print("Stoped")
                queue.put("Done")
                continue
            clean_values = []
            command = "jumpcutter"
            for arg in spec.args:
                val = spec.annotations[arg](values[arg + "_input"])
                clean_values.append(val)
                command += f" --{arg} {val}"
            process = Process(
                target=run_command,
                args=(clean_values, queue),
            )
            window["command_line"].update(command)
            window["Start"].update("Stop", button_color=("white", "red"))
            print("Tarefa Iniciada!")
            window.refresh()
            process.daemon = True
            process.start()

    window.close()


def FText(
    text,
    default=None,
    tooltip=None,
    input_size=(20, 1),
    text_size=(20, 1),
):
    return [
        sg.Text(text, size=text_size, justification="r", tooltip=tooltip),
        sg.Input(
            default_text=default,
            key=text + "_input",
            size=input_size,
            tooltip=tooltip,
        ),
    ]


def run_command(values, queue: Queue):
    sys.stdout = RedirectProcess(queue)
    sys.stderr = sys.stdout
    cli.main(*values)
    queue.put("Done")


if __name__ == "__main__":
    gui()
