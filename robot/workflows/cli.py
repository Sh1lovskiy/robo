"""Command line interface for robot workflows."""

from __future__ import annotations

import argparse

from robot.controller import RobotController
from utils.cli import Command, CommandDispatcher
from utils.config import Config
from utils.logger import Logger
from utils.lmdb_storage import LmdbStorage

from .record import JsonPoseSaver, DBPoseSaver, PoseRecorder, CameraManager, DBFrameSaver
from .path import PathRunner


def _add_record_args(parser: argparse.ArgumentParser) -> None:
    Config.load()
    parser.add_argument("--ip", default=Config.get("robot.ip"), help="Robot IP")
    parser.add_argument(
        "--captures_dir",
        default=Config.get("path_saver.captures_dir", "captures"),
        help="Directory for saved poses",
    )
    parser.add_argument("--drag", action="store_true", help="Enable drag teaching mode")
    parser.add_argument("--db_path", default="robot_data.lmdb", help="LMDB database path")
    parser.add_argument("--use-db", action="store_true", help="Use LMDB storage")


def _run_record(args: argparse.Namespace) -> None:
    storage = LmdbStorage(args.db_path)
    saver = DBPoseSaver(storage) if args.use_db else JsonPoseSaver()
    recorder = PoseRecorder(
        controller=RobotController(rpc=args.ip),
        saver=saver,
        captures_dir=args.captures_dir,
        drag=args.drag,
    )
    recorder.run()


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    Config.load()
    parser.add_argument("--ip", default=Config.get("robot.ip"), help="Robot IP")
    parser.add_argument("--db_path", default="robot_data.lmdb", help="LMDB database with poses")


def _run_path(args: argparse.Namespace) -> None:
    storage = LmdbStorage(args.db_path)
    runner = PathRunner(
        controller=RobotController(rpc=args.ip),
        camera_mgr=CameraManager(),
        frame_saver=DBFrameSaver(storage),
        storage=storage,
    )
    runner.run()


def _add_restart_args(parser: argparse.ArgumentParser) -> None:
    Config.load()
    parser.add_argument("--ip", default=Config.get("robot.ip"), help="Robot IP")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between reconnects")
    parser.add_argument("--attempts", type=int, default=3, help="Reconnect attempts")


def _run_restart(args: argparse.Namespace) -> None:
    controller = RobotController(rpc=args.ip)
    ok = controller.restart(ip_address=args.ip, delay=args.delay, attempts=args.attempts)
    if ok:
        controller.logger.info("Robot restart completed successfully")
    else:
        controller.logger.error("Failed to restart robot")


def create_cli() -> CommandDispatcher:
    return CommandDispatcher(
        "Robot workflows",
        [
            Command("record", _run_record, _add_record_args, "Record robot poses"),
            Command("run", _run_path, _add_run_args, "Execute path and capture"),
            Command("restart", _run_restart, _add_restart_args, "Restart robot connection"),
        ],
    )


def main() -> None:
    logger = Logger.get_logger("robot.workflows")
    create_cli().run(logger=logger)
