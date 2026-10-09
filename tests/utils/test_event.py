import unittest
from unittest.mock import patch

import torch

from deep_ep import EventHandle
from deep_ep.utils import event as event_module
from deep_ep.utils.event import EventOverlap


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class TestEventOverlap(unittest.TestCase):

    def test_other_streams_wait_for_hook_completion(self):
        for release_handle in (False, True):
            with self.subTest(release_handle=release_handle):
                value = torch.zeros(256, dtype=torch.int32, device='cuda')
                epilogue_stream = torch.cuda.Stream()
                reader_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
                overlap = EventOverlap(EventHandle())
                epilogue_done = torch.cuda.Event()
                calls = []
                torch.cuda.synchronize()

                def epilogue(value=value, calls=calls, epilogue_done=epilogue_done):
                    calls.append(1)
                    torch.cuda._sleep(200_000_000)
                    value.fill_(37)
                    epilogue_done.record()
                    return value

                overlap.register_hook_after_wait(epilogue)
                snapshots = []
                try:
                    with torch.cuda.stream(epilogue_stream):
                        self.assertIs(overlap.current_stream_wait(), value)
                    # Require the deliberately delayed write to still be pending.
                    # Otherwise this run cannot exercise the cross-stream race.
                    self.assertFalse(epilogue_done.query(), 'epilogue delay already completed')
                    for index, stream in enumerate(reader_streams):
                        with torch.cuda.stream(stream):
                            self.assertIsNone(
                                overlap.current_stream_wait(release_handle=release_handle and index == len(reader_streams) - 1))
                            snapshots.append(value.clone())
                finally:
                    for stream in reader_streams:
                        stream.synchronize()
                    epilogue_stream.synchronize()
                self.assertEqual(len(calls), 1)
                for snapshot in snapshots:
                    self.assertEqual(snapshot.unique().tolist(), [37])
                self.assertEqual(overlap.event is None, release_handle)

    def test_plain_wait_does_not_record_an_epilogue_event(self):
        original = EventHandle()
        overlap = EventOverlap(original)
        with patch.object(event_module, 'EventHandle', wraps=EventHandle) as capture:
            self.assertIsNone(overlap.wait())
            self.assertIsNone(overlap.current_stream_wait(release_handle=True))
            capture.assert_not_called()
        self.assertIsNone(overlap.event)

    def test_hook_failure_can_be_retried(self):
        overlap = EventOverlap(EventHandle())
        attempts = []

        def epilogue():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError('hook failed')
            return 19

        overlap.register_hook_after_wait(epilogue)
        with self.assertRaisesRegex(RuntimeError, 'hook failed'):
            overlap.wait()
        self.assertEqual(overlap.wait(), 19)
        self.assertIsNone(overlap.wait())
        self.assertEqual(len(attempts), 2)

    def test_context_manager_waits_for_prior_hook(self):
        value = torch.zeros(256, dtype=torch.int32, device='cuda')
        epilogue_stream, reader_stream = torch.cuda.Stream(), torch.cuda.Stream()
        overlap = EventOverlap(EventHandle())
        epilogue_done = torch.cuda.Event()
        torch.cuda.synchronize()

        def epilogue():
            torch.cuda._sleep(200_000_000)
            value.fill_(53)
            epilogue_done.record()

        overlap.register_hook_after_wait(epilogue)
        try:
            with torch.cuda.stream(epilogue_stream):
                overlap.wait()
            self.assertFalse(epilogue_done.query(), 'epilogue delay already completed')
            with torch.cuda.stream(reader_stream):
                with overlap(release_handle=True):
                    pass
                snapshot = value.clone()
        finally:
            reader_stream.synchronize()
            epilogue_stream.synchronize()
        self.assertEqual(snapshot.unique().tolist(), [53])
        self.assertIsNone(overlap.event)

    def test_hook_completion_under_cuda_graph(self):
        value = torch.zeros(256, dtype=torch.int32, device='cuda')
        snapshot = torch.empty_like(value)
        capture_stream, epilogue_stream, reader_stream = (torch.cuda.Stream() for _ in range(3))
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()

        with torch.cuda.graph(graph, stream=capture_stream):
            value.zero_()
            overlap = EventOverlap(EventHandle())

            def epilogue():
                torch.cuda._sleep(2_000_000)
                value.fill_(71)

            overlap.register_hook_after_wait(epilogue)
            with torch.cuda.stream(epilogue_stream):
                overlap.wait()
            with torch.cuda.stream(reader_stream):
                overlap.current_stream_wait()
                snapshot.copy_(value)
                reader_done = EventHandle()
            # Join the reader and writer back into the capture stream.
            reader_done.current_stream_wait()
            with torch.cuda.stream(epilogue_stream):
                epilogue_done = EventHandle()
            epilogue_done.current_stream_wait()

        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(snapshot.unique().tolist(), [71])


if __name__ == '__main__':
    unittest.main()
