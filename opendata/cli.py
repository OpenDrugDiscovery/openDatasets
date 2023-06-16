import numpy as np
import typer
import fsspec
import logging
import datamol as dm
from opendata.collections.peptides import NaturalPeptides


# disable openff-toolkit warnings
logging.getLogger("openff.toolkit").setLevel(logging.ERROR)

# CLI
app = typer.Typer(
    help="ODD cli for dataset collection and conformers enumeration + optimization", add_completion=True, pretty_exceptions_enable=False
)

@app.command("peptides", help="Compute the geometry optimization for all molecules in the given input_file")
def peptides(
    input_file: str = typer.Argument(..., help="Path to the input file"),
    server_info_file: str = typer.Option(default=None, help="Path to the server info file"),
):
    cd = NaturalPeptides(input_file=input_file, server_info_file=server_info_file)
    cd.submit()



if __name__ == "__main__":
    app()
    
