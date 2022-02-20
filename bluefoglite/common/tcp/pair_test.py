# Copyright 2021 Bluefog Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import socket
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from bluefoglite import BlueFogLiteEventError

from bluefoglite.common.handle_manager import HandleManager
from bluefoglite.common.store import InMemoryStore
from bluefoglite.common.tcp.buffer import SpecifiedBuffer, UnspecifiedBuffer
from bluefoglite.common.tcp.eventloop import EventLoop
from bluefoglite.common.tcp.pair import Pair, SocketFullAddress
from bluefoglite.testing.util import multi_thread_help


# Remember to make sure that two sides of pair are binded to some port before they
# call the connect. Here store served as the coordinating role.
@pytest.fixture(name="array_list")
def fixture_array_list():
    n = 9
    return [np.arange(n), np.zeros((n,)).astype(int)]


@pytest.fixture(name="empty_address")
def fixture_empty_address():
    return SocketFullAddress(
        addr=("localhost", 0),
        sock_family=socket.AF_INET,
        sock_type=socket.SOCK_STREAM,
        sock_protocol=socket.IPPROTO_IP,
    )


def test_connect(empty_address):
    store = InMemoryStore()

    def listen_connect_close(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            full_address=empty_address.copy(),
        )
        store.set(rank, pair.self_address)
        pair.connect(store.get(1 - rank))

        pair.close()
        event_loop.close()

    errors = multi_thread_help(size=2, fn=listen_connect_close)

    for error in errors:
        raise error


def _build_sbuf_from_array(array):
    mock_context = MagicMock()
    return SpecifiedBuffer(mock_context, array.data.cast("c"), array.nbytes)


@pytest.mark.parametrize("reverse_send_recv", [True, False])
def test_send_recv_array(empty_address, array_list, reverse_send_recv):
    store = InMemoryStore()

    def send_recv_array(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            full_address=empty_address.copy(),
        )
        store.set(rank, pair.self_address)
        pair.connect(store.get(1 - rank))

        hm = HandleManager.getInstance()
        send_rank, recv_rank = (0, 1) if reverse_send_recv else (1, 0)

        if rank == send_rank:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[0])
            pair.send(buf, handle, nbytes=buf.buffer_length, offset=0)
            hm.wait(handle=handle)
        elif rank == recv_rank:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[1])
            pair.recv(buf, handle, nbytes=buf.buffer_length, offset=0)
            hm.wait(handle=handle)
            np.testing.assert_allclose(array_list[1], array_list[0])
        pair.close()
        event_loop.close()

    errors = multi_thread_help(size=2, fn=send_recv_array)

    for error in errors:
        raise error


def test_send_recv_array_multiple(empty_address, array_list):
    store = InMemoryStore()

    def send_recv_array(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            full_address=empty_address.copy(),
        )
        store.set(rank, pair.self_address)
        pair.connect(store.get(1 - rank))

        hm = HandleManager.getInstance()
        send_rank, recv_rank = (0, 1)

        if rank == send_rank:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[0])
            pair.send(buf, handle, nbytes=buf.buffer_length, offset=0)
            hm.wait(handle=handle)
        elif rank == recv_rank:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[1])
            pair.recv(buf, handle, nbytes=buf.buffer_length, offset=0)
            hm.wait(handle=handle)
            np.testing.assert_allclose(array_list[1], array_list[0])
        pair.close()
        event_loop.close()

    for _ in range(20):
        errors = multi_thread_help(size=2, fn=send_recv_array)
        store.reset()
        for error in errors:
            raise error


@pytest.mark.parametrize("reverse_send_recv", [True, False])
def test_send_recv_obj(empty_address, reverse_send_recv):
    store = InMemoryStore()

    def send_recv_obj(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            full_address=empty_address.copy(),
        )
        store.set(rank, pair.self_address)
        pair.connect(store.get(1 - rank))

        hm = HandleManager.getInstance()
        send_rank, recv_rank = (0, 1) if reverse_send_recv else (1, 0)

        data = b"jisdnoldf"
        if rank == send_rank:
            handle = hm.allocate()
            buf = SpecifiedBuffer(MagicMock(), memoryview(data).cast("c"), len(data))
            pair.send(buf, handle, nbytes=buf.buffer_length, offset=0)
            hm.wait(handle=handle)
        elif rank == recv_rank:
            handle = hm.allocate()
            ubuf = UnspecifiedBuffer(MagicMock())
            pair.recv(ubuf, handle, nbytes=-1, offset=0)
            hm.wait(handle=handle)
            assert ubuf.data == data
        pair.close()
        event_loop.close()

    errors = multi_thread_help(size=2, fn=send_recv_obj)

    for error in errors:
        raise error


@pytest.mark.parametrize("reverse_send_recv", [True, False])
def test_send_after_peer_close(empty_address, array_list, reverse_send_recv):
    store = InMemoryStore()

    def send_after_peer_close(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            full_address=empty_address.copy(),
        )
        store.set(rank, pair.self_address)
        pair.connect(store.get(1 - rank))

        hm = HandleManager.getInstance()
        send_rank, recv_rank = (0, 1) if reverse_send_recv else (1, 0)

        if rank == send_rank:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[0])
            time.sleep(0.5)  # wait to send
            try:
                pair.send(buf, handle, nbytes=buf.buffer_length, offset=0)
                hm.wait(handle=handle)
            except BlueFogLiteEventError:
                # Encounter error: [Errno 32] Broken pipe
                pass
            pair.close()
        elif rank == recv_rank:
            # Close immediately
            pair.close()

        event_loop.close()

    errors = multi_thread_help(size=2, fn=send_after_peer_close)

    for error in errors:
        raise error


@pytest.mark.parametrize("reverse_send_recv", [True, False])
def test_close_before_send_finish(empty_address, array_list, reverse_send_recv, caplog):
    store = InMemoryStore()

    def close_before_send_finish(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            full_address=empty_address.copy(),
        )
        store.set(rank, pair.self_address)
        pair.connect(store.get(1 - rank))

        hm = HandleManager.getInstance()
        send_rank, recv_rank = (0, 1) if reverse_send_recv else (1, 0)

        # Close immediate after Send
        if rank == send_rank:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[0])
            pair.send(buf, handle, nbytes=buf.buffer_length, offset=0)
            pair.close()
            # Close it withour wait sending.
            with caplog.at_level(logging.WARNING):
                hm.wait(handle=handle)

            assert len(caplog.record_tuples) == 1
            assert caplog.record_tuples[0][:2] == ("BFL_LOGGER", logging.WARNING)
            assert (
                "Unfinished send/recv after pair is closed"
                in caplog.record_tuples[0][2]
            )
        elif rank == recv_rank:
            time.sleep(0.1)
            pair.close()

        event_loop.close()

    errors = multi_thread_help(size=2, fn=close_before_send_finish)

    for error in errors:
        raise error


@pytest.mark.parametrize("reverse_send_recv", [True, False])
def test_close_before_recv_finish(empty_address, array_list, reverse_send_recv, caplog):
    store = InMemoryStore()

    def close_before_recv_finish(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            full_address=empty_address.copy(),
        )
        store.set(rank, pair.self_address)
        pair.connect(store.get(1 - rank))

        hm = HandleManager.getInstance()
        send_rank, recv_rank = (0, 1) if reverse_send_recv else (1, 0)

        if rank == send_rank:
            time.sleep(0.1)
            pair.close()
        elif rank == recv_rank:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[0])
            pair.recv(buf, handle, nbytes=buf.buffer_length, offset=0)
            pair.close()
            # Close it without wait receiving.
            with caplog.at_level(logging.WARNING):
                hm.wait(handle=handle)

            assert len(caplog.record_tuples) == 1
            assert caplog.record_tuples[0][:2] == ("BFL_LOGGER", logging.WARNING)
            assert (
                "Unfinished send/recv after pair is closed"
                in caplog.record_tuples[0][2]
            )

        event_loop.close()

    errors = multi_thread_help(size=2, fn=close_before_recv_finish)

    for error in errors:
        raise error
