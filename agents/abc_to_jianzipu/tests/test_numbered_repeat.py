import unittest
from copy import deepcopy

from agents.ToolRuntime.runtime import (
    related_repeat_phrases, repeat_content_warnings, repeat_anchor_warnings,
    render_edit_preview,
)
from agents.abc_to_jianzipu.teacher_trajectory import render_public_prompt
from agents.abc_to_jianzipu.repeat_materializer import OMITTED_PLACEHOLDER


def item(tones, start=0, event_start=0, phrase='p0001', score='test'):
    notes = [{'index':start+i, 'event_index':event_start+i, 'jianpu':tone,
              'abc':'', 'duration':'四分'} for i, tone in enumerate(tones)]
    actions = [{'source_index':n['index'], 'jianzi_text':'散挑一弦'} for n in notes]
    return {'score_key':score, 'phrase_id':phrase, 'input':{
        'metadata':{'tonic':'1=C'}, 'normalized_tuning':{},
        'event_range':{'start':start}, 'notes_without_jianzi':notes},
        'baseline_plan':{'actions':deepcopy(actions)}, 'reference_plan':{'actions':actions}}


class NumberedRepeatTests(unittest.TestCase):
    def setup_repeat(self, head, current, command='从20再作'):
        prior = item(head, event_start=20)
        now = item(current, start=len(head), event_start=100, phrase='p0002')
        actions = {n['index']:{'source_index':n['index'], 'jianzi_text':
                   command if i==0 else OMITTED_PLACEHOLDER}
                   for i,n in enumerate(now['input']['notes_without_jianzi'])}
        return now, actions, {('test','p0001'):prior}

    def test_numeric_repeat_uses_public_event_index(self):
        for command in ['从20再作', '从20（音序）再作', '从音序20再做']:
            now, actions, history = self.setup_repeat(['1','2','3'], ['1','2','3'], command)
            self.assertEqual(repeat_content_warnings(now,actions,historical=history), [])
            self.assertEqual(repeat_anchor_warnings(now,actions,historical=history), [])

    def test_all_omitted_notes_are_compared_not_just_five(self):
        now,actions,history=self.setup_repeat(['1','2','3','4','5','6'], ['1','2','3','4','5','7'])
        warnings=repeat_content_warnings(now,actions,historical=history)
        self.assertEqual(warnings[0]['code'],'repeat_numbered_pitch_mismatch')
        self.assertEqual(warnings[0]['compared_events'],6)
        preview=render_edit_preview(list(actions.values()),[],True,[],now,warnings)
        self.assertIn('指定音序再作序列不匹配',preview)

    def test_future_or_missing_index_warns(self):
        for command in ['从0再作', '从100再作', '从999再作']:
            now,actions,history=self.setup_repeat(['1','2'],['1','2'],command)
            self.assertEqual(repeat_content_warnings(now,actions,historical=history)[0]['code'],
                             'repeat_numbered_start_missing')

    def test_blank_repeat_tails_still_warn(self):
        now,actions,history=self.setup_repeat(['1','2'],['1','2'])
        actions[len(history[('test','p0001')]['input']['notes_without_jianzi'])+1]['jianzi_text']=''
        codes={w['code'] for w in repeat_content_warnings(now,actions,historical=history)}
        self.assertIn('repeat_omission_marker_missing',codes)

    def test_invalid_repeat_does_not_request_omission_placeholders(self):
        for command, current, expected in [
            ('从999再作', ['1','2'], 'repeat_numbered_start_missing'),
            ('从20再作', ['1','3'], 'repeat_numbered_pitch_mismatch'),
            ('从ㄱ再作', ['1','2'], 'missing_repeat_anchor'),
        ]:
            now, actions, history = self.setup_repeat(['1','2'], current, command)
            actions[3]['jianzi_text'] = ''
            content = repeat_content_warnings(now, actions, historical=history)
            codes = {w['code'] for w in content + repeat_anchor_warnings(now, actions, historical=history)}
            self.assertIn(expected, codes)
            self.assertNotIn('repeat_omission_marker_missing', codes)

    def test_valid_anchor_still_requests_omission_placeholders(self):
        now, actions, history = self.setup_repeat(['1','2'], ['1','2'], '从ㄱ再作')
        history[('test','p0001')]['reference_plan']['actions'][0]['jianzi_text'] = '[再作标记ㄱ]散挑一弦'
        actions[3]['jianzi_text'] = ''
        self.assertIn('repeat_omission_marker_missing', {
            w['code'] for w in repeat_content_warnings(now, actions, historical=history)})

    def test_nine_notes_retrieve_and_render_related_phrase(self):
        tones=['1','2','3','4','5','6','7','1̇','2̇']
        prior=item(tones,event_start=20)
        now=item(tones,start=9,event_start=100,phrase='p0002')
        history={('test','p0001'):prior}
        matches=related_repeat_phrases(now,history)
        self.assertEqual(len(matches),1)
        self.assertEqual(matches[0]['matched_from_event'],20)
        prompt=render_public_prompt(now,'fingering_agent',historical=history)
        self.assertIn('只读相关段 p0001',prompt)
        self.assertIn('历史音序20',prompt)
        self.assertIn('从n再作',prompt)

    def test_eight_notes_wrong_octave_or_other_score_do_not_retrieve(self):
        tones=['1','2','3','4','5','6','7','1̇','2̇']
        now=item(tones,start=9,phrase='p0002')
        for head,score in [(tones[:8],'test'),(tones[:-1]+['2'],'test'),(tones,'other')]:
            history={(score,'p0001'):item(head,score=score)}
            self.assertEqual(related_repeat_phrases(now,history),[])
            self.assertNotIn('只读相关段',render_public_prompt(now,'fingering_agent',historical=history))

    def test_match_across_earlier_phrase_boundary(self):
        tones=['1','2','3','4','5','6','7','1̇','2̇']
        history={('test','p0001'):item(tones[:4]),
                 ('test','p0002'):item(tones[4:],start=4,event_start=4,phrase='p0002')}
        now=item(tones,start=9,event_start=9,phrase='p0003')
        self.assertEqual([m['phrase_id'] for m in related_repeat_phrases(now,history)],['p0001'])

    def test_multiple_matches_select_only_earliest_phrase(self):
        tones=['1','2','3','4','5','6','7','1̇','2̇']
        history={('test','p0002'):item(tones,start=9,event_start=9,phrase='p0002'),
                 ('test','p0001'):item(tones)}
        now=item(tones,start=18,event_start=18,phrase='p0003')
        self.assertEqual([m['phrase_id'] for m in related_repeat_phrases(now,history)],['p0001'])
        prompt=render_public_prompt(now,'fingering_agent',historical=history)
        self.assertEqual(prompt.count('【只读相关段'),1)
        self.assertIn('只读相关段 p0001',prompt)

    def test_rest_or_missing_range_breaks_retrieval(self):
        tones=['1','2','3','4','5','6','7','1̇','2̇']
        now=item(tones,start=20,phrase='p0003')
        prior=item(tones)
        prior['input']['notes_without_jianzi'][4]['abc']='z4'
        self.assertEqual(related_repeat_phrases(now,{('test','p0001'):prior}),[])
        history={('test','p0001'):item(tones[:4]),
                 ('test','p0002'):item(tones[4:],start=5,event_start=4,phrase='p0002')}
        self.assertEqual(related_repeat_phrases(now,history),[])

    def test_previous_related_phrase_is_not_duplicated(self):
        tones=['1','2','3','4','5','6','7','1̇','2̇']
        prior=item(tones)
        now=item(tones,start=9,event_start=9,phrase='p0002')
        now['input']['phrase_handoff']={'phrase_id':'p0002','previous_phrase':{
            'phrase_id':'p0001','notes':prior['input']['notes_without_jianzi'],
            'actions':prior['reference_plan']['actions']}}
        prompt=render_public_prompt(now,'fingering_agent',historical={('test','p0001'):prior})
        self.assertIn('只读前一段 p0001',prompt)
        self.assertNotIn('只读相关段 p0001',prompt)
        self.assertIn('相关匹配',prompt)

    def test_missing_anchor_warning_suggests_numeric_repeat(self):
        now,actions,history=self.setup_repeat(['1','2'],['1','2'],'从ㄱ再作')
        warnings=repeat_anchor_warnings(now,actions,historical=history)
        self.assertIn('从n再作',render_edit_preview(list(actions.values()),[],True,[],now,warnings))

    def test_mostly_empty_first_match_is_skipped_for_next_populated_match(self):
        tones=['1','2','3','4','5','6','7','1̇','2̇']
        first=item(tones)
        for action in first['reference_plan']['actions'][:5]:action['jianzi_text']=''
        second=item(tones,start=9,event_start=9,phrase='p0002')
        now=item(tones,start=18,event_start=18,phrase='p0003')
        self.assertEqual([m['phrase_id'] for m in related_repeat_phrases(now,
            {('test','p0001'):first,('test','p0002'):second})],['p0002'])

    def test_empty_or_omitted_notation_does_not_make_a_useful_match(self):
        tones=['1','2','3','4','5','6','7','1̇','2̇']
        for empty in ['',None,OMITTED_PLACEHOLDER]:
            prior=item(tones)
            for action in prior['reference_plan']['actions'][:5]:action['jianzi_text']=empty
            now=item(tones,start=9,phrase='p0002')
            self.assertEqual(related_repeat_phrases(now,{('test','p0001'):prior}),[])

    def test_matched_span_mostly_empty_even_when_whole_phrase_is_populated(self):
        tones=['1','2','3','4','5','6','7','1̇','2̇']
        prior=item(tones+['7']*10)
        for action in prior['reference_plan']['actions'][:5]:action['jianzi_text']=''
        self.assertEqual(related_repeat_phrases(item(tones,start=19,phrase='p0002'),
            {('test','p0001'):prior}),[])

    def test_exact_half_nonempty_is_allowed(self):
        tones=['1','2','3','4','5','6','7','1̇','2̇','3̇']
        prior=item(tones)
        for action in prior['reference_plan']['actions'][:5]:action['jianzi_text']=''
        self.assertEqual(len(related_repeat_phrases(item(tones,start=10,phrase='p0002'),
            {('test','p0001'):prior})),1)

    def test_related_block_precedes_previous_block(self):
        tones=['1','2','3','4','5','6','7','1̇','2̇']
        prior=item(tones);previous=item(['7','6'],start=9,event_start=9,phrase='p0002')
        now=item(tones,start=11,event_start=11,phrase='p0003')
        now['input']['phrase_handoff']={'phrase_id':'p0003','previous_phrase':{
            'phrase_id':'p0002','notes':previous['input']['notes_without_jianzi'],
            'actions':previous['reference_plan']['actions']}}
        prompt=render_public_prompt(now,'fingering_agent',historical={('test','p0001'):prior,('test','p0002'):previous})
        self.assertLess(prompt.index('【只读相关段'),prompt.index('【只读前一段'))
        self.assertLess(prompt.index('【只读前一段'),prompt.index('【当前段'))


if __name__=='__main__':
    unittest.main()
