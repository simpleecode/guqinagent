import signal
import subprocess
import unittest
from unittest.mock import Mock, patch

from train.scripts.run_vllm_eval_queue import idle_server_indices, stop_server_group


class ReplicaReleaseTests(unittest.TestCase):
    def test_first_worker_finish_does_not_release_shared_replica(self):
        # Workers 0 and 4 share replica 0; 4 is still running.
        self.assertEqual(idle_server_indices([1, 2, 3, 4], 4, set()), [])
        self.assertEqual(idle_server_indices([1, 2, 3], 4, set()), [0])

    def test_other_replicas_remain_alive(self):
        self.assertEqual(idle_server_indices([0, 2], 4, set()), [1, 3])

    def test_released_replicas_are_not_released_twice(self):
        self.assertEqual(idle_server_indices([2], 4, {0, 1}), [3])
        self.assertEqual(idle_server_indices([], 4, {0, 1, 2, 3}), [])

    def test_successful_shutdown_closes_log_and_uses_owned_group(self):
        server = Mock(pid=12345); log = Mock()
        with patch('train.scripts.run_vllm_eval_queue.os.killpg',
                   side_effect=[None, ProcessLookupError()]) as kill:
            stop_server_group(server, log)
        self.assertEqual(kill.call_args_list[0].args, (12345, signal.SIGINT))
        log.close.assert_called_once()

    def test_exited_launcher_still_stops_engine_descendants(self):
        server = Mock(pid=12345); server.poll.return_value = 0; log = Mock()
        with patch('train.scripts.run_vllm_eval_queue.os.killpg',
                   side_effect=[None, None, None, ProcessLookupError()]) as kill:
            stop_server_group(server, log)
        self.assertIn((12345, signal.SIGTERM), [call.args for call in kill.call_args_list])
        log.close.assert_called_once()

    def test_timeout_escalates_and_waits_for_sigkill(self):
        server = Mock(pid=12345); log = Mock()
        server.wait.side_effect = [subprocess.TimeoutExpired('server', 30),
                                   subprocess.TimeoutExpired('server', 10), 0]
        with patch('train.scripts.run_vllm_eval_queue.os.killpg',
                   side_effect=[None, None, None, ProcessLookupError()]) as kill:
            stop_server_group(server, log)
        self.assertEqual([call.args[1] for call in kill.call_args_list],
                         [signal.SIGINT, signal.SIGTERM, signal.SIGKILL, 0])
        log.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
