import uuid
from sqlalchemy import Column, Integer, String, Float, Boolean, Index, ForeignKey
from sqlalchemy.dialects.postgresql import JSON, UUID
from qcfractal.storage_sockets.models import MsgpackExt, Base
from sqlalchemy.orm import relationship


class MolecularSystemORM(Base):
    """
    The molecule DB collection is managed by pymongo, so far
    """

    __tablename__ = "molecular_system"
    id = Column(UUID(as_uuid=False), primary_key=True, default=uuid.uuid4, unique=True)
    name = Column(String)
    type= Column(String)
    inchikey = Column(String)
    smiles = Column(String)
    count = Column(Integer)
    provenance = Column(String)
    molecular_formula = Column(String)

    conformers = Column(MsgpackExt)
    optimization_trajectories = Column(MsgpackExt)

   

class ConformerORM(Base):
    """
    The conformer DB collection is managed by pymongo, so far
    """

    __tablename__ = "conformer"

    id = Column(UUID(as_uuid=False), primary_key=True, default=uuid.uuid4, unique=True)
    system_id = Column(UUID(as_uuid=False), ForeignKey("molecular_system.id", ondelete="cascade"), primary_key=True)
    molecular_formula = Column(String)

    # TODO - hash can be stored more efficiently (ie, byte array)
    molecule_hash = Column(String)

    # Required data
    schema_name = Column(String)
    schema_version = Column(Integer, default=2)
    symbols = Column(MsgpackExt, nullable=False)
    geometry = Column(MsgpackExt, nullable=False)

    # Molecule data
    name = Column(String, default="")
    identifiers = Column(JSON)
    comment = Column(String)
    molecular_charge = Column(Float, default=0)
    molecular_multiplicity = Column(Integer, default=1)

    # Atom data
    masses = Column(MsgpackExt)
    real = Column(MsgpackExt)
    atom_labels = Column(MsgpackExt)
    atomic_numbers = Column(MsgpackExt)
    mass_numbers = Column(MsgpackExt)

    # Fragment and connection data
    connectivity = Column(JSON)
    fragments = Column(MsgpackExt)
    fragment_charges = Column(JSON)  # Column(ARRAY(Float))
    fragment_multiplicities = Column(JSON)  # Column(ARRAY(Integer))

    # Orientation
    fix_com = Column(Boolean, default=False)
    fix_orientation = Column(Boolean, default=False)
    fix_symmetry = Column(String)

    # Extra
    provenance = Column(JSON)
    extras = Column(JSON)

    __table_args__ = (Index("ix_molecule_hash", "molecule_hash", unique=False),)


class TrajectoryConformerORM(Base):
    """
    The trajectory is a collection of conformers that are ordered by optimization steps 
    """
    __tablename__ = "trajectory_conformer"
    trajectory_id = Column(UUID(as_uuid=False), ForeignKey("trajectory.id", ondelete="cascade"), primary_key=True)
    conformer_id = Column(UUID(as_uuid=False), ForeignKey("conformer.id", ondelete="cascade"), primary_key=True)
    position = Column(Integer, )


class TrajectoryORM(Base):
    """
    The trajectory is a collection of conformers that are ordered by optimization steps 
    """
    __tablename__ = "trajectory"
    id = Column(UUID(as_uuid=False), primary_key=True, default=uuid.uuid4, unique=True)
    conformers = relationship('TrajectoryConformerORM', uselist=True, backref='tr')
