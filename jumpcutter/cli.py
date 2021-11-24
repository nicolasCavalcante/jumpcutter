import typer

app = typer.Typer()


@app.command()
def hello(name: str = typer.Option("")):
    if name != "":
        typer.echo(f"Hello {name}")
    else:
        typer.echo("Hello World!")


@app.command()
def bye(name: str = typer.Option("")):
    if name != "":
        typer.echo(f"Bye {name}")
    else:
        typer.echo("Goodbye!")


if __name__ == "__main__":
    app()
