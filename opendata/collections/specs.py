from qcportal.singlepoint.record_models import QCSpecification
from qcportal.singlepoint.record_models import SinglepointDriver


gaz_phase_dft = QCSpecification(
    program = "psi4",
    driver = SinglepointDriver.gradient,
    method = "wb97m-d3bj",
    basis = "def2-tzvppd",
    keywords = {"wcombine": False, 
                # "solvent": "none", # "water",
                "scf_properties": [
                    "MBIS_CHARGES",
                    "WIBERG_LOWDIN_INDICES",
                    "MAYER_INDICES",
                    "LOWDIN_CHARGES",
                    "DIPOLE",
                    "QUADRUPOLE",
                ]},
    protocols = {"wavefunction": "all"}
)

implicit_water_dft = QCSpecification(
    program = "psi4",
    driver = SinglepointDriver.gradient,
    method = "wb97m-d3bj",
    basis = "def2-tzvppd",
    keywords = {"wcombine": False, 
                "pcm": True,
                "pcm_solvent": "Water",
                "scf_properties": [
                    "MBIS_CHARGES",
                    "WIBERG_LOWDIN_INDICES",
                    "MAYER_INDICES",
                    "LOWDIN_CHARGES",
                    "DIPOLE",
                    "QUADRUPOLE",
                ]},
    protocols = {"wavefunction": "all"}
)

gaz_phase_se = QCSpecification(
        program="xtb",
        method="GFN2-xTB",
        driver = SinglepointDriver.gradient,
        basis=None,
        keywords={
            "wcombine": False,
            "scf_type": "df",
            "electronic_temperature": 300.0,
            "max_iterations": 50,
            "solvent": "none", # "water",
            "maxiter": 200
        }
    )

implicit_water_se = QCSpecification(
        program="xtb",
        method="GFN2-xTB",
        driver = SinglepointDriver.gradient,
        basis=None,
        keywords={
            "wcombine": False,
            "scf_type": "df",
            "electronic_temperature": 300.0,
            "max_iterations": 50,
            "solvent": "water",
            "maxiter": 200
        }
    )

def get_specs(name):
    SPECS = dict(
        gaz_phase_dft=gaz_phase_dft,
        implicit_water_dft=implicit_water_dft,
        gaz_phase_se=gaz_phase_se,
        implicit_water_se=implicit_water_se,
    )

    if name not in SPECS:
        raise Exception("Undefined spec configuration")
    return SPECS[name]
