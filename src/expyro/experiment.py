from __future__ import annotations

import inspect
from datetime import datetime
from functools import update_wrapper
from pathlib import Path
from typing import Generic, TypeVar, TypeAlias, Callable, Iterator, Mapping, Iterable, get_type_hints, Any

from matplotlib import pyplot as plt

from src.expyro.serialization import load_config, load_result, dump_config, dump_result, has_config, has_result
from src.expyro.util import unique_new_path

T_Config = TypeVar("T_Config")
T_Result = TypeVar("T_Result")

Procedure: TypeAlias = Callable[[T_Config], T_Result]
Plot: TypeAlias = plt.Figure | Mapping[str, plt.Figure]
Artist: TypeAlias = Callable[[T_Config, T_Result], Plot]


class ProcedureSignature(Generic[T_Config, T_Result]):
    config_name: str | None
    config_cls: type[T_Config] | None
    result_cls: type[T_Result] | None

    def __init__(self, procedure: Procedure[T_Config, T_Result]):
        type_hints = get_type_hints(procedure)
        self.result_cls = type_hints.pop("return") if "return" in type_hints else None

        if len(type_hints) == 1:
            self.config_name, self.config_cls = type_hints.popitem()
        else:
            self.config_name = None
            self.config_cls = None


class Experiment(Generic[T_Config, T_Result]):
    procedure: Procedure[T_Config, T_Result]
    signature: ProcedureSignature[T_Config, T_Result]
    directory: Path
    name: str
    plotters: list[Plotter]
    config_options: dict[str, T_Config]

    def __init__(self, procedure: Procedure[T_Config, T_Result], directory: Path, name: str):
        assert inspect.isfunction(procedure), "Procedure must be a function."
        assert len(inspect.signature(procedure).parameters) == 1, "Procedure must have exactly one parameter."

        self.procedure = procedure
        self.signature = ProcedureSignature(procedure)
        self.directory = directory
        self.name = name
        self.plotters = []
        self.config_options = {}

        update_wrapper(self, procedure)

    def __folder(self, location: Path | str | None = None) -> Path:
        if location is None:
            return self.directory / self.name
        if isinstance(location, str):
            return self.directory / self.name / location
        return location

    def __make_folder(self) -> Path:
        folder = self.__folder() / datetime.now().strftime("%Y-%m-%d %H-%M-%S.%f")
        folder.mkdir(parents=True, exist_ok=False)
        return folder

    def extend_plots(self, plots: Iterable[Plotter]):
        self.plotters.extend(plots)

    def extend_config_options(self, options: Mapping[str, T_Config]):
        self.config_options |= options

    def config(self, location: Path | str) -> T_Config:
        location = self.__folder(location)
        return load_config(location)

    def result(self, location: Path | str) -> T_Result:
        location = self.__folder(location)
        return load_result(location)

    def plot(self, config: T_Config, result: T_Result, folder: Path):
        for plotter in self.plotters:
            plotter(config, result, folder)

    def cli(self) -> tuple[Any, list[str]]:
        from .command import cli
        return cli(self)

    def __call__(self, config: T_Config) -> Run[T_Config, T_Result]:
        folder = self.__make_folder()

        dump_config(config, folder)
        result = self.procedure(config)
        dump_result(result, folder)

        run = Run(config, result, folder, self)

        if self.plotters:
            run.plot()

        return run

    def __contains__(self, item: Path | str) -> bool:
        location = self.__folder(item)
        return has_config(location) and has_result(location)

    def __getitem__(self, item: Path | str) -> Run[T_Config, T_Result]:
        location = self.__folder(item)
        config = self.config(item)
        result = self.result(item)
        return Run(config, result, location, self)

    def __iter__(self) -> Iterator[Run[T_Config, T_Result]]:
        base_folder = self.__folder()
        return (self[item] for item in base_folder.iterdir() if item.is_dir() and item in self)


class Run(Generic[T_Config, T_Result]):
    config: T_Config
    result: T_Result
    location: Path
    __experiment: Experiment[T_Config, T_Result]

    def __init__(self, config: T_Config, result: T_Result, location: Path, experiment_: Experiment[T_Config, T_Result]):
        self.config = config
        self.result = result
        self.location = location
        self.__experiment = experiment_

    def __make_plot_folder(self) -> Path:
        folder = unique_new_path(self.location / "plots")
        folder.mkdir(parents=True, exist_ok=False)
        return folder

    def reproduce(self) -> Run[T_Config, T_Result]:
        return self.__experiment(self.config)

    def plot(self) -> Path:
        folder = self.__make_plot_folder()
        self.__experiment.plot(self.config, self.result, folder)
        return folder

    def rename(self, name: str) -> Run[T_Config, T_Result]:
        new_location = unique_new_path(self.location.parent / name)
        self.location.rename(new_location)
        return Run(self.config, self.result, new_location, self.__experiment)


class Plotter(Generic[T_Config, T_Result]):
    artist: Artist
    file_format: str
    save_kwargs: dict
    show: bool

    def __init__(self, artist: Artist, file_format: str, show: bool, **kwargs):
        self.artist = artist
        self.file_format = file_format
        self.save_kwargs = kwargs
        self.show = show

    def __dump_figure(self, figure: plt.Figure, folder: Path, name: str):
        figure.savefig(folder / f"{name}.{self.file_format}", **self.save_kwargs)

        if self.show:
            plt.show()

        plt.close(figure)

    def __call__(self, config: T_Config, result: T_Result, directory: Path):
        result = self.artist(config, result)

        if isinstance(result, Mapping):
            directory = directory / self.artist.__name__
            directory.mkdir(exist_ok=True)

            for name, figure in result.items():
                self.__dump_figure(figure, directory, name)
        else:
            self.__dump_figure(result, directory, self.artist.__name__)


ProcedureDecorator: TypeAlias = Callable[[Procedure[T_Config, T_Result]], Experiment[T_Config, T_Result]]
ExperimentDecorator: TypeAlias = Callable[[Experiment[T_Config, T_Result]], Experiment[T_Config, T_Result]]


def experiment(root_directory: Path, name: str | None = None) -> ProcedureDecorator:
    def decorator(procedure: Procedure[T_Config, T_Result]) -> Experiment[T_Config, T_Result]:
        nonlocal name
        name = procedure.__name__ if name is None else name
        return Experiment(procedure, root_directory, name)

    return decorator


def plot(*artists: Artist, file_format: str = "pdf", show: bool = False, **kwargs) -> ExperimentDecorator:
    def decorator(experiment_: Experiment[T_Config, T_Result]) -> Experiment[T_Config, T_Result]:
        experiment_.extend_plots([Plotter(artist, file_format, show, **kwargs) for artist in artists])
        return experiment_

    return decorator


def config_option(name: str, config: T_Config) -> ExperimentDecorator:
    def decorator(experiment_: Experiment[T_Config, T_Result]) -> Experiment[T_Config, T_Result]:
        experiment_.extend_config_options({name: config})
        return experiment_

    return decorator


def config_options(options: Mapping[str, T_Config]) -> ExperimentDecorator:
    def decorator(experiment_: Experiment[T_Config, T_Result]) -> Experiment[T_Config, T_Result]:
        experiment_.extend_config_options(options)
        return experiment_

    return decorator
