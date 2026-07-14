"""Split-inference seam: serialize an intermediate activation at a configured exit index
and hand it off to a transport.

JOLT owns this seam; the volunteer extension (cross-device split inference) implements a
real ``Transport`` that ships the bytes to another device. The default ``IdentityTransport``
keeps everything in-process so the rest of the pipeline runs unchanged, and ``FileTransport``
round-trips through disk for offline testing. The contract is deliberately byte-oriented so
the transport makes no assumptions about the model or the device.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

import torch


def serialize_activation(tensor: torch.Tensor) -> bytes:
    """Serialize an activation tensor to bytes (moved to CPU first)."""
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue()


def deserialize_activation(payload: bytes, device: torch.device | str = "cpu") -> torch.Tensor:
    """Inverse of :func:`serialize_activation`."""
    buffer = io.BytesIO(payload)
    tensor = torch.load(buffer, map_location="cpu", weights_only=True)
    return tensor.to(device)


class Transport(Protocol):
    """A transport moves serialized activation bytes from the head device to the tail device.

    The extension supplies a network/socket implementation. As long as ``send`` returns the
    same bytes that were handed off (after whatever round trip it performs), the split run is
    numerically identical to the unsplit run.
    """

    def send(self, payload: bytes, *, exit_index: int) -> bytes: ...


class IdentityTransport:
    """No-op transport (in-process). Default, so split inference is a pure pass-through."""

    def send(self, payload: bytes, *, exit_index: int) -> bytes:
        return payload


class FileTransport:
    """Round-trip the payload through a file. Useful for offline tests and capture."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def send(self, payload: bytes, *, exit_index: int) -> bytes:
        path = self.directory / f"activation_exit_{exit_index}.pt"
        path.write_bytes(payload)
        return path.read_bytes()


def handoff(
    tensor: torch.Tensor,
    *,
    exit_index: int,
    transport: Transport | None = None,
) -> torch.Tensor:
    """Serialize ``tensor`` at ``exit_index``, push it through ``transport``, return the
    reconstructed tensor on the original device. With the default identity transport this is
    an exact round trip."""
    transport = transport or IdentityTransport()
    payload = serialize_activation(tensor)
    returned = transport.send(payload, exit_index=exit_index)
    return deserialize_activation(returned, device=tensor.device)
