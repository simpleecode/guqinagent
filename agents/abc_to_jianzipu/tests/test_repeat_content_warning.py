import unittest
from agents.ToolRuntime.runtime import repeat_content_warnings, render_edit_preview
from agents.abc_to_jianzipu.repeat_materializer import OMITTED_PLACEHOLDER


class RepeatContentTests(unittest.TestCase):
    def setup_rows(self, head, target, texts=None):
        prefix_notes = [{'index': i, 'event_index': i, 'jianpu': tone, 'abc': '', 'duration': '四分'}
                        for i, tone in enumerate(head)]
        current_notes = [{'index': i+len(head), 'event_index': i+len(head),
                          'jianpu': tone, 'abc': '', 'duration': '八分'}
                         for i, tone in enumerate(target)]
        def row(notes):
            return {'score_key': 'test', 'input': {'metadata': {'tonic': '1=C'},
                    'event_range': {'start': notes[0]['index']}, 'notes_without_jianzi': notes}}
        prior = row(prefix_notes)
        prior['reference_plan'] = {'actions': []}
        item = row(current_notes)
        texts = texts or ['从头再作']+[OMITTED_PLACEHOLDER]*(len(target)-1)
        actions = {n['index']: {'source_index': n['index'], 'jianzi_text': text}
                   for n, text in zip(current_notes, texts)}
        return item, actions, {('test', 'p0001'): prior}

    def warnings(self, *args, **kwargs):
        item, actions, history = self.setup_rows(*args, **kwargs)
        return repeat_content_warnings(item, actions, historical=history)

    def test_matches_only_first_five_not_later_variation_or_duration(self):
        self.assertEqual(self.warnings(['1','2','3','4','5','6'], ['1','2','3','4','5','7']), [])

    def test_wrong_octave_is_a_mismatch(self):
        warnings = self.warnings(['1','2','3','4','5'], ['1̇','2','3','4','5'])
        self.assertEqual(warnings[0]['code'], 'repeat_from_head_pitch_mismatch')
        self.assertEqual(warnings[0]['compared_events'], 5)

    def test_short_tail_checks_available_sounds(self):
        warnings = self.warnings(['1','2','3','4','5'], ['6','7'])
        self.assertEqual(warnings[0]['compared_events'], 2)

    def test_omitted_sounded_rows_need_placeholder_but_not_sustain_rest_or_bar(self):
        item, actions, history = self.setup_rows(['1','2','3','4','5'],
            ['1','2','－（延音）','0（休止）','|','3'], ['从头再作','','','','',''])
        item['input']['notes_without_jianzi'][4]['abc'] = '|'
        warnings = repeat_content_warnings(item, actions, historical=history)
        missing = next(w for w in warnings if w['code']=='repeat_omission_marker_missing')
        self.assertEqual(missing['missing_source_indices'], [6,10])
        self.assertEqual(sum(w['code']=='repeat_omission_marker_missing' for w in warnings), 1)

    def test_attached_instruction_starts_comparison_on_next_note(self):
        self.assertEqual(self.warnings(['1','2','3','4','5'], ['6','1','2','3','4','5'],
            ['大指七徽勾四弦从头再作']+[OMITTED_PLACEHOLDER]*5), [])

    def test_either_score_or_actual_section_head_can_match(self):
        item, actions, history = self.setup_rows(['1','2','3','4','5','6','7','1','2','3'],
                                                  ['6','7','1','2','3'])
        prior = next(iter(history.values()))
        for n in prior['input']['notes_without_jianzi']:
            n['section'] = {'number': 1 if n['index']<5 else 2, 'start': n['index']==5}
        for n in item['input']['notes_without_jianzi']:
            n['section'] = {'number': 2, 'start': False}
        self.assertEqual(repeat_content_warnings(item, actions, historical=history), [])

    def test_phrase_head_is_not_a_candidate(self):
        warnings = self.warnings(['1','2','3','4','5'], ['6','7','1','2','3'])
        self.assertTrue(any(w['code']=='repeat_from_head_pitch_mismatch' for w in warnings))

    def test_missing_history_or_unknown_pitch_does_not_guess(self):
        item, actions, _ = self.setup_rows(['1','2','3','4','5'], ['6','7'])
        self.assertEqual(repeat_content_warnings(item, actions), [])
        item, actions, history = self.setup_rows(['1','?','3','4','5'], ['6','7'])
        self.assertEqual(repeat_content_warnings(item, actions, historical=history), [])

    def test_inherited_repeat_tail_across_phrase_boundary(self):
        item, actions, history = self.setup_rows(['1','2','3','4','5'], ['6','7','1','2','3'])
        prior = next(iter(history.values()))
        prior['input']['notes_without_jianzi'] += item['input']['notes_without_jianzi'][:2]
        prior['reference_plan']['actions'] = [actions[5], actions[6]]
        item['input']['notes_without_jianzi'] = item['input']['notes_without_jianzi'][2:]
        item['input']['event_range']['start'] = 7
        actions = {i:a for i,a in actions.items() if i>=7}
        actions[7]['jianzi_text'] = ''
        warnings = repeat_content_warnings(item, actions, historical=history)
        self.assertEqual({w['code'] for w in warnings},
                         {'repeat_from_head_pitch_mismatch','repeat_omission_marker_missing'})
        self.assertTrue(all(w['source_index']==7 for w in warnings))

    def test_real_attack_ends_omission_interval(self):
        warnings = self.warnings(['1','2','3','4','5'], ['1','2','3','4'],
                                 ['从头再作', OMITTED_PLACEHOLDER,'大指七徽勾四弦',''])
        self.assertEqual(warnings, [])

    def test_section_boundary_ends_repeat_tail(self):
        item, actions, history = self.setup_rows(['1','2','3','4','5'], ['1','2','3'],
                                                 ['从头再作', '', ''])
        item['input']['notes_without_jianzi'][2]['section'] = {'number': 2, 'start': True}
        warnings = repeat_content_warnings(item, actions, historical=history)
        self.assertEqual(warnings[0]['missing_source_indices'], [6])

    def test_current_sealed_reference_does_not_supply_repeat_evidence(self):
        item, actions, history = self.setup_rows(['1','2','3','4','5'], ['6','7'])
        expected = repeat_content_warnings(item, actions, historical=history)
        item['reference_plan'] = {'actions': [{'source_index': 5, 'jianzi_text': '合法指法'}]}
        self.assertEqual(repeat_content_warnings(item, actions, historical=history), expected)

    def test_chord_component_order_does_not_matter(self):
        item, actions, history = self.setup_rows(['1'], ['1̇'])
        prior_note = next(iter(history.values()))['input']['notes_without_jianzi'][0]
        prior_note.update(abc='[Cc]', jianpu_alt='1̇')
        item['input']['notes_without_jianzi'][0].update(abc='[cC]', jianpu_alt='1')
        self.assertEqual(repeat_content_warnings(item, actions, historical=history), [])

    def test_new_warnings_are_rendered_on_unmodified_rows(self):
        item, actions, history = self.setup_rows(['1','2','3','4','5'], ['6','7'], ['从头再作',''])
        warnings = repeat_content_warnings(item, actions, historical=history)
        preview = render_edit_preview(list(actions.values()), [], True, [], item, warnings)
        self.assertIn(':warning:从头再作序列与曲首或段首不匹配', preview)
        self.assertIn(':warning:再作省略音缺少省略标记', preview)
        self.assertIn('音序6', preview)


if __name__ == '__main__':
    unittest.main()
