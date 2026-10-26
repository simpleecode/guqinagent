import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from train.scripts.visualize_completed_eval import publish,validate,safe_name
from ABC_J.scripts.visualize_running_eval_score import stage_trace


class ManagedViewerTests(unittest.TestCase):
    def test_request_messages_are_opt_in_and_source_is_unchanged(self):
        stage={'ok':True,'trace':[{'round':1,'raw_output':'answer',
                'request_messages':[{'role':'user','content':'REQUEST_SENTINEL'}]}]}
        self.assertNotIn('本轮实际请求 messages',stage_trace(stage,'test'))
        self.assertNotIn('REQUEST_SENTINEL',stage_trace(stage,'test'))
        self.assertIn('REQUEST_SENTINEL',stage_trace(stage,'test',include_request_messages=True))
        self.assertEqual(stage['trace'][0]['request_messages'][0]['content'],'REQUEST_SENTINEL')

    def fixture(self, score='ScoreA'):
        source={'sample_id':score+'-p0001','score_key':score,'phrase_id':'p0001',
                'input_sha256':'abc','runtime_item':{'input':{'metadata':{'score_title':'测试'}}}}
        prediction={'sample_id':source['sample_id'],'score_key':score,'input_sha256':'abc',
                    'protocol_valid':True,'base':{'ok':True},'guqinizer':{'ok':True}}
        return source,prediction

    def setup_data(self, root):
        s,r=self.fixture();a=root/'input.jsonl';b=root/'pred.jsonl'
        a.write_text(json.dumps(s)+'\n');b.write_text(json.dumps(r)+'\n')
        return a,b,r

    def test_storage_dark_index_and_retained_pdf(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a,b,r=self.setup_data(root)
            publish(a,b,'two_stage',output_root=root)
            target=root/'two_stage/ScoreA';pdf=target/'ScoreA.pdf';pdf.write_bytes(b'unchanged pdf')
            with patch('train.scripts.visualize_completed_eval._retire',side_effect=lambda p:p.rename(root/('old-'+p.name))):
                publish(a,b,'two_stage',output_root=root)
            self.assertEqual(pdf.read_bytes(),b'unchanged pdf')
            self.assertIn('background:#181818',(root/'two_stage/index.html').read_text())
            self.assertIn('ScoreA/ScoreA.html',(root/'two_stage/index.html').read_text())

    def test_changed_prediction_invalidates_pdf_and_zip(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a,b,r=self.setup_data(root);publish(a,b,'two_stage',output_root=root)
            method=root/'two_stage';(method/'ScoreA/ScoreA.pdf').write_bytes(b'old')
            (method/'all_scores_pdf.zip').write_bytes(b'old zip')
            r['change']=True;b.write_text(json.dumps(r)+'\n')
            with patch('train.scripts.visualize_completed_eval._retire',side_effect=lambda p:p.rename(root/('old-'+p.name))):
                publish(a,b,'two_stage',output_root=root)
            self.assertFalse((method/'ScoreA/ScoreA.pdf').exists())
            self.assertFalse((method/'all_scores_pdf.zip').exists())
            self.assertNotIn('ScoreA.pdf',(method/'pdf_index.html').read_text())

    def test_render_failure_leaves_current_result_untouched(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a,b,r=self.setup_data(root);publish(a,b,'two_stage',output_root=root)
            target=root/'two_stage/ScoreA/predictions.jsonl';old=target.read_bytes()
            r['change']=True;b.write_text(json.dumps(r)+'\n')
            with patch('train.scripts.visualize_completed_eval.render_score',side_effect=RuntimeError('render failed')):
                with self.assertRaises(RuntimeError):publish(a,b,'two_stage',output_root=root,render=True)
            self.assertEqual(target.read_bytes(),old)
            self.assertFalse(list((root/'two_stage').glob('.visualize-*')))

    def test_invalid_inputs_and_paths(self):
        s,r=self.fixture()
        for a,b in [([s],[]),([s,s],[r]),([s],[r,r]),([s],[dict(r,input_sha256='bad')])]:
            with self.assertRaises(ValueError):validate(a,b)
        for path in ['../x','/tmp','.work','a/b']:
            with self.assertRaises(ValueError):safe_name(path)

    def test_method_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a,b,r=self.setup_data(root);(root/'archive').mkdir()
            (root/'two_stage').symlink_to(root/'archive',target_is_directory=True)
            with self.assertRaises(ValueError):publish(a,b,'two_stage',output_root=root)


if __name__=='__main__':unittest.main()
