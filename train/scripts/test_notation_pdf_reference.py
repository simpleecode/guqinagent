import unittest
from pathlib import Path
from unittest.mock import patch
from copy import deepcopy
from ABC_J.scripts.visualize_running_eval_score import render_notation_pdf


class ReferenceRowTests(unittest.TestCase):
    def test_reference_is_separate_display_row(self):
        source=[{'sample_id':'s-p0001','phrase_id':'p0001','input':{
            'notes_without_jianzi':[{'index':0,'jianpu':'1','abc':'C','event_index':0}]}}]
        predictions={'s-p0001':{'guqinizer':{'plan':{'actions':[
            {'source_index':0,'jianzi_text':'散勾三弦'}]}}}}
        original=deepcopy((source,predictions))
        with patch('reportlab.pdfgen.canvas.Canvas') as canvas:
            render_notation_pdf(source,predictions,'s','title',{},Path('/tmp/mock-only.pdf'),
                                reference_by_id={'s-p0001':{0:'散挑四弦'}})
            labels=[c.args[-1] for c in canvas.return_value.drawCentredString.call_args_list]
            self.assertIn('模型生成',labels)
            self.assertIn('原标注',labels)
            self.assertIn('散勾三弦',labels)
            self.assertIn('散挑四弦',labels)
        self.assertEqual((source,predictions),original)

    def test_missing_reference_fails_explicitly(self):
        with patch('reportlab.pdfgen.canvas.Canvas'):
            with self.assertRaisesRegex(ValueError,'Missing PDF annotation'):
                render_notation_pdf([{'sample_id':'s','input':{}}],{'s':{}},'s','',{},
                                    Path('/tmp/mock-only.pdf'),include_pending=True,reference_by_id={})
