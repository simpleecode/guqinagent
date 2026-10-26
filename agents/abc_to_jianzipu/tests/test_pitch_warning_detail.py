import unittest
from agents.ToolRuntime.runtime import (
    RealToolRuntime,
    inherited_gesture_warning_text,
    pitch_mismatch_warning_text, public_pitch_warning_source_indices,
    public_plan_warning_messages, render_edit_preview, validate_jianzi_only,
)
from agents.abc_to_jianzipu.reference_parser import AUDIT
from agents.abc_to_jianzipu.teacher_trajectory import _render_phrase_lines


class PitchWarningDetailTests(unittest.TestCase):
    def test_taoqi_warning_distinguishes_wrong_finger_from_wrong_string(self):
        missing = inherited_gesture_warning_text({
            'code': 'taoqi_without_thumb_origin', 'string': 5,
            'previous_finger': '名指',
        })
        self.assertIn('前段实际使用的是名指', missing)
        mismatch = inherited_gesture_warning_text({
            'code': 'taoqi_without_thumb_origin', 'string': 5,
            'previous_string': 2, 'previous_finger': '大指',
        })
        self.assertIn('此前在同弦(5弦)以左手大指按弦', mismatch)
        self.assertIn('前段按音在2弦', mismatch)

    def test_inherited_gesture_warning_reaches_public_prompt(self):
        notes = [
            {'index': 42, 'event_index': 30, 'jianpu': '1̣', 'abc': 'F,4', 'duration': '四分'},
            {'index': 43, 'event_index': 31, 'jianpu': '1̣', 'abc': 'F,4', 'duration': '四分'},
        ]
        actions = [{'source_index': 42, 'jianzi_text': '散挑三弦'},
                   {'source_index': 43, 'jianzi_text': '注下徽外'}]
        item = {'score_key': 'S35eJqgV', 'phrase_id': 'p0002',
                'input': {'metadata': {'tonic': '1=F', 'tonic_degree1_midi': 65},
                          'normalized_tuning': {'open_midi': [48, 50, 53, 55, 57, 60, 62]},
                          'notes_without_jianzi': notes},
                'baseline_plan': {'actions': actions}}
        _, report = validate_jianzi_only(item, [], require_complete=False)
        self.assertTrue(any(w['source_index'] == 43 and
                            w['code'] == 'open_string_unplayable_gesture'
                            for w in report['warnings']))
        messages = public_plan_warning_messages(item)
        self.assertIn('散音后不能承接注下徽外', messages[43][0])

    def test_kneeling_taoqi_hui_warning_uses_public_prefix(self):
        notes = [
            {'index': 0, 'event_index': 0, 'jianpu': '1', 'abc': 'C4', 'duration': '四分'},
            {'index': 1, 'event_index': 1, 'jianpu': '2', 'abc': 'D4', 'duration': '四分'},
        ]
        actions = [{'source_index': 0, 'jianzi_text': '大指九徽挑六弦'},
                   {'source_index': 1, 'jianzi_text': '跪指八徽滔起'}]
        item = {'score_key': 'Stest', 'phrase_id': 'p0001',
                'input': {'metadata': {'tonic': '1=C'},
                          'normalized_tuning': {'open_midi': [48, 50, 53, 55, 57, 60, 62]},
                          'notes_without_jianzi': notes},
                'baseline_plan': {'actions': actions}}
        _, report = validate_jianzi_only(item, [], require_complete=False)
        preview = render_edit_preview(actions, [{'source_index': 1, 'patch_type': 'SET_JIANZI_TEXT'}],
                                      True, [], item=item, warnings=report['warnings'])
        self.assertIn(':warning:跪指滔起只适用于七徽及更靠近岳山的徽位（请使用名指滔起）',
                      preview)

    def test_li_count_warning_identifies_only_the_short_sweep(self):
        text = pitch_mismatch_warning_text({
            'li_pitch_count_mismatch': True,
            'actual_midis': [85, 83, 81], 'expected_midis': [81, 81],
        })
        self.assertIn('“历”发出3音，本处只有2个连续目标起音', text)
        self.assertIn('多出的音没有对应目标', text)
        self.assertNotIn('小节线', text)

    def test_integer_fractional_and_open(self):
        for mode, hui, expected in [('stopped', 10, '3弦十徽按音'), ('stopped', 7.6, '七徽六分'),
                                     ('harmonic', 7, '泛音'), ('open', None, '3弦散音')]:
            text = pitch_mismatch_warning_text({'resolved_pitches': [
                {'string': 3, 'hui': hui, 'mode': mode, 'midi': 67}], 'expected_midis': [65]})
            self.assertIn(expected, text)
            if mode == 'open':
                self.assertNotIn('徽', text)
            elif hui == 10:
                self.assertIn('十徽', text)
                self.assertNotIn('分', text)
            self.assertIn('G4', text)
            self.assertIn('目标F4', text)

    def test_audit_retains_inherited_position(self):
        report = AUDIT.audit({'metadata': {'tonic': '1=C'}, 'open_midi': [48,50,53,55,57,60,62],
            'notes': [{'index': 0, 'jianpu': '1', 'jianzi': '名指十徽挑六弦'},
                      {'index': 1, 'jianpu': '1', 'jianzi': '名指十徽滔起'}]}, 50)
        sound = report['details'][1]['resolved_pitches'][0]
        self.assertEqual((sound['string'], sound['hui'], sound['mode']), (6,10,'stopped'))

    def test_trailing_harmonic_stop_does_not_erase_sound(self):
        r = AUDIT.audit({'metadata': {'tonic': '1=C'}, 'open_midi': [48,50,53,55,57,60,62],
                        'notes': [{'index':0,'jianpu':'1','jianzi':'泛音七徽挑三弦泛止'}]},50)
        self.assertEqual(r['details'][0]['resolved_pitches'][0]['mode'], 'harmonic')

    def test_li_mismatch_points_to_covered_note_and_edit_origin(self):
        notes = [
            {'index': index, 'event_index': index - 102, 'jianpu': '3̇',
             'abc': 'a', 'duration': '八分'}
            for index in (103, 104, 105)
        ]
        actions = [
            {'source_index': 103, 'jianzi_text': '泛起食指四徽挑五弦'},
            {'source_index': 104, 'jianzi_text': '历五四弦'},
            {'source_index': 105, 'jianzi_text': ''},
        ]
        item = {'score_key': 'test', 'phrase_id': 'p0001',
                'input': {'metadata': {'tonic': '1=F', 'tonic_degree1_midi': 65},
                          'normalized_tuning': {'open_midi': [48, 50, 53, 55, 57, 60, 62]},
                          'notes_without_jianzi': notes},
                'baseline_plan': {'actions': actions}}
        _, report = validate_jianzi_only(item, [], require_complete=False)
        pitch_warnings = [warning for warning in report['warnings']
                          if warning['code'] == 'jianzi_pitch_mismatch']
        self.assertEqual(len(pitch_warnings), 1)
        self.assertEqual((pitch_warnings[0]['source_index'],
                          pitch_warnings[0]['origin_source_index']), (105, 104))
        self.assertEqual(pitch_warnings[0]['actual_midis'], [79.0])
        messages = public_plan_warning_messages(item)
        prompt_rows = _render_phrase_lines(
            notes, actions,
            pitch_warning_source_indices=public_pitch_warning_source_indices(item),
            warning_messages_by_source=messages,
        )
        row = next(line for line in prompt_rows.splitlines() if line.startswith('3｜'))
        self.assertIn('[空]:warning:音高不匹配', row)
        self.assertIn('4弦四徽泛音音高为G5', row)
        self.assertIn('请修改音序2', row)
        self.assertEqual(row.count(':warning:音高不匹配'), 1)

    def test_li_crosses_barline_in_public_preview(self):
        notes = [
            {'index': 343, 'event_index': 263, 'jianpu': '3', 'abc': 'A2', 'duration': '四分'},
            {'index': 344, 'jianpu': '|', 'abc': '|', 'duration': '小节线'},
            {'index': 345, 'event_index': 264, 'jianpu': '3', 'abc': 'A2', 'duration': '四分'},
        ]
        actions = [
            {'source_index': 343, 'jianzi_text': '大指六徽四分历五四弦'},
            {'source_index': 345, 'jianzi_text': ''},
        ]
        item = {'score_key': 'S35eJqgV', 'phrase_id': 'p0017',
                'input': {'metadata': {'tonic': '1=F'},
                          'normalized_tuning': {'open_midi': [48, 50, 53, 55, 57, 60, 62]},
                          'notes_without_jianzi': notes},
                'baseline_plan': {'actions': actions}}
        _, report = validate_jianzi_only(item, [], require_complete=False)
        warnings = [warning for warning in report['warnings']
                    if warning['code'] == 'jianzi_pitch_mismatch']
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['source_index'], 343)
        self.assertFalse(warnings[0].get('li_pitch_count_mismatch'))
        preview = render_edit_preview(actions,
                                      [{'source_index': 343, 'patch_type': 'SET_JIANZI_TEXT'}],
                                      True, [], item=item, warnings=report['warnings'])
        self.assertIn('5弦六徽四分按音音高为B4（MIDI 71.21）', preview)
        self.assertNotIn('多出的音没有对应目标', preview)

    def test_taoqi_different_string_warning_in_edit_preview(self):
        notes = [
            {'index': 343, 'event_index': 263, 'jianpu': '3', 'abc': 'A2', 'duration': '四分'},
            {'index': 345, 'event_index': 264, 'jianpu': '3', 'abc': 'A2', 'duration': '四分'},
            {'index': 346, 'event_index': 265, 'jianpu': '2', 'abc': 'G', 'duration': '四分'},
            {'index': 347, 'event_index': 266, 'jianpu': '1', 'abc': 'F', 'duration': '四分'},
        ]
        actions = [
            {'source_index': 343, 'jianzi_text': '大指七徽挑五弦'},
            {'source_index': 345, 'jianzi_text': ''},
            {'source_index': 346, 'jianzi_text': '注下七徽六分'},
            {'source_index': 347, 'jianzi_text': '名指七徽三弦滔起'},
        ]
        item = {'score_key': 'S35eJqgV', 'phrase_id': 'p0017',
                'input': {'metadata': {'tonic': '1=F'},
                          'normalized_tuning': {'open_midi': [48, 50, 53, 55, 57, 60, 62]},
                          'notes_without_jianzi': notes},
                'baseline_plan': {'actions': actions}}
        _, report = validate_jianzi_only(item, [], require_complete=False)
        warning = next(w for w in report['warnings']
                       if w['source_index'] == 347
                       and w['code'] == 'taoqi_without_thumb_origin')
        self.assertEqual((warning['previous_string'], warning['string']), (5, 3))
        preview = render_edit_preview(actions,
                                      [{'source_index': 347, 'patch_type': 'SET_JIANZI_TEXT'}],
                                      True, [], item=item, warnings=report['warnings'])
        self.assertIn('266｜1｜', preview)
        self.assertIn(':warning:滔起前置条件不满足', preview)
        self.assertIn('同弦(3弦)', preview)

        # Check the actual public edit_plan envelope, not only its formatter.
        item['baseline_plan']['actions'][-1]['jianzi_text'] = ''
        tool_reply = RealToolRuntime(item, {}).invoke(
            'edit_plan', {'jianzi_rows': [[266, '名指七徽三弦滔起']]})
        self.assertTrue(tool_reply['ok'])
        self.assertIn('266｜1｜', tool_reply['result']['text'])
        self.assertIn(':warning:滔起前置条件不满足', tool_reply['result']['text'])


if __name__ == '__main__':
    unittest.main()
