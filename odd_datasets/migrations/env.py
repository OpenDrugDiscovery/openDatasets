import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context
from qcfractal.storage_sockets.models import Base

def database_uri(config, db_name) -> str:

    uri = "postgresql://"
    if config.get("user") is not None:
        uri += f"{config.get('user')}"

        if config.get('passwd') is not None:
            uri += ':'+config.get('passwd')

        uri += "@"

    if config.get('port') is not None:
        uri += f"{config.get('host')}:{config.get('port')}/{db_name}"
    else:
        uri += f"{config.get('host')}/{db_name}"

    return uri


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
# target_metadata = None
target_metadata = Base.metadata
compare_type = True

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

# Overwrite the ini-file sqlalchemy.url path
uri = context.get_x_argument(as_dictionary=True).get("uri")
print(uri)
os.environ["ODD_POSTGRES_DB_USER"] = "prudencio"
# os.environ["ODD_POSTGRES_DB_PASSWORD"] = "odd1234"
os.environ["ODD_POSTGRES_DB_HOST"] = "localhost"
# os.environ["ODD_POSTGRES_DB_PORT"] = "5432"
uri_args = dict(
    user=os.environ.get('ODD_POSTGRES_DB_USER', None),
    passwd=os.environ.get('ODD_POSTGRES_DB_PASSWORD', None),
    host=os.environ.get('ODD_POSTGRES_DB_HOST', "localhost"),
    port=os.environ.get('ODD_POSTGRES_DB_PORT', None),
)
uri = database_uri(uri_args, "qcfractal_default")
print("uri", uri)
# exit(1)
config.set_main_option("sqlalchemy.url", uri)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()



