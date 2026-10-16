import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ABC_J.scripts import generate_teacher_tool_trajectories as teacher


class RateLimitError(Exception):
    status_code = 429


class APITimeoutError(Exception):
    pass


class TeacherRequestRetryTests(unittest.TestCase):
    def options(self):
        return {'model':'glm-5.3', 'messages':[
            {'role':'assistant','content':'previous successful tool call'},
            {'role':'user','content':'previous tool receipt'},
        ]}

    def call(self, messages, options):
        return teacher.create_teacher_response(
            SimpleNamespace(messages=messages),options,
            trajectory_id='test-p0001',stage='guqinization',round_number=4)

    @patch.object(teacher.time, 'sleep')
    @patch.object(teacher.random, 'uniform', return_value=0)
    def test_429_retries_identical_request_and_keeps_prior_messages(self, jitter, sleep):
        response=object()
        messages=Mock();messages.create.side_effect=[RateLimitError('limited'),response]
        options=self.options();before=deepcopy(options)
        self.assertIs(self.call(messages,options),response)
        self.assertEqual(options,before)
        self.assertEqual(messages.create.call_count,2)
        for call in messages.create.call_args_list:
            self.assertIs(call.kwargs['messages'],options['messages'])
        sleep.assert_called_once_with(5.0)

    @patch.object(teacher.time, 'sleep')
    def test_nontransient_error_is_not_request_retried(self,sleep):
        messages=Mock();messages.create.side_effect=ValueError('bad payload')
        with self.assertRaises(ValueError):self.call(messages,self.options())
        self.assertEqual(messages.create.call_count,1);sleep.assert_not_called()

    @patch.dict(teacher.os.environ, {'GLM_REQUEST_MAX_ATTEMPTS':'2'})
    @patch.object(teacher.time, 'sleep')
    def test_exhaustion_is_distinguished_from_stage_validation_failure(self,sleep):
        messages=Mock();messages.create.side_effect=RateLimitError('limited')
        with self.assertRaises(teacher.RequestRetryExhausted):self.call(messages,self.options())
        self.assertEqual(messages.create.call_count,2)

    def test_exhausted_request_does_not_restart_generate_one(self):
        failures=[]
        with patch.object(teacher,'generate_one',side_effect=teacher.RequestRetryExhausted('limited')) as generate:
            with self.assertRaises(teacher.RequestRetryExhausted):
                teacher.generate_with_retries(
                    SimpleNamespace(messages=Mock()),'glm-5.3',
                    {'trajectory_id':'test-p0001'},'guqinization',[],{},24,3,
                    objective='review_noop',failure_traces=failures)
        self.assertEqual(generate.call_count,1)
        self.assertEqual(len(failures),1)

    @patch.object(teacher.time, 'sleep')
    def test_timeout_retries_current_request(self,sleep):
        messages=Mock();messages.create.side_effect=[APITimeoutError('timeout'),'success']
        self.assertEqual(self.call(messages,self.options()),'success')
        self.assertEqual(messages.create.call_count,2)

    @patch.object(teacher.random, 'uniform', return_value=0)
    @patch.object(teacher.time, 'sleep')
    def test_retry_after_and_trace_preserve_failed_and_successful_requests(self,sleep,jitter):
        error=RateLimitError('limited');error.response=SimpleNamespace(headers={'retry-after':'15'})
        response=SimpleNamespace(model_dump=lambda **kwargs:{'id':'success'})
        underlying=Mock();underlying.create.side_effect=[error,response]
        trace=[]
        recording=teacher._RecordingClient(SimpleNamespace(messages=underlying),trace)
        self.assertIs(teacher.create_teacher_response(recording,self.options(),
            trajectory_id='test-p0001',stage='guqinization',round_number=4),response)
        sleep.assert_called_once_with(15.0)
        self.assertEqual(len(trace),2)
        self.assertIn('error',trace[0]);self.assertIn('response',trace[1])


if __name__=='__main__':unittest.main()
