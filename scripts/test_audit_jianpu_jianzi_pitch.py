import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("audit_jianpu_jianzi_pitch.py")
SPEC = importlib.util.spec_from_file_location("audit_pitch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AuditPitchTests(unittest.TestCase):
    def test_fu_requires_live_walk_origin(self):
        def fu_warnings(texts):
            return [(w['index'], w['code']) for w in MODULE.inherited_gesture_warnings({
                'open_midi': self.opens,
                'notes': [{'index': i, 'jianzi': text} for i, text in enumerate(texts)]
            }) if w['code'] in {'fu_without_walk_origin', 'repeated_fu_without_walk'}]
        self.assertEqual(fu_warnings(['大指七徽勾三弦', '名指七徽勾二弦', '复']),
                         [(2, 'fu_without_walk_origin')])
        self.assertEqual(fu_warnings(['大指九徽勾三弦', '退', '复']), [])
        self.assertEqual(fu_warnings(['大指九徽勾三弦', '注下十徽', '猱', '复']), [])
        self.assertEqual(fu_warnings(['大指九徽勾三弦', '退', '勾三弦', '复']),
                         [(3, 'fu_without_walk_origin')])
        self.assertEqual(fu_warnings(['大指九徽勾三弦', '退', '复', '复']),
                         [(3, 'repeated_fu_without_walk')])
        self.assertEqual(fu_warnings(['大指十徽勾二弦绰上九徽', '复']), [])

    def test_slide_attack_requires_explicit_left_finger(self):
        for text in ('绰上七徽勾五弦', '绰上七徽挑六弦', '注下九徽挑六弦',
                     '泛止绰上七徽勾五弦'):
            warnings = MODULE.inherited_gesture_warnings({
                'open_midi': self.opens, 'notes': [{'index': 1, 'jianzi': text}]})
            self.assertIn('slide_attack_missing_left_finger', [w['code'] for w in warnings], text)
        for text in ('绰上七徽', '绰大指七徽勾5弦', '注名指九徽挑六弦',
                     '大指十徽勾二弦绰上九徽'):
            warnings = MODULE.inherited_gesture_warnings({
                'open_midi': self.opens, 'notes': [{'index': 1, 'jianzi': text}]})
            self.assertNotIn('slide_attack_missing_left_finger', [w['code'] for w in warnings], text)

    def test_fake_open_ornament_is_warned_without_inherited_state(self):
        for text in ('散猱', '散吟', '散撞', '散音猱', '散猱二弦'):
            warnings = MODULE.inherited_gesture_warnings({
                'open_midi': self.opens,
                'notes': [{'index': 1, 'jianzi': text}],
            })
            self.assertEqual([w['code'] for w in warnings], ['open_attack_ornament'], text)

    def test_positioned_pluck_inherits_stopped_string_and_updates_hui(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi('大指五徽勾六弦', self.opens, context)
        for text, hui in [('挑四徽六分', 4.6), ('挑四徽三分', 4.3)]:
            actual, error = MODULE.parse_jianzi(text, self.opens, context)
            expected, _ = MODULE.position_pitch(6, hui, self.opens, 'stopped')
            self.assertIsNone(error)
            self.assertAlmostEqual(actual[0], expected)
            self.assertEqual(context['active_left_string'], 6)
            self.assertEqual(context['active_left_hui'], hui)
        warnings = MODULE.walk_motion_warnings({'open_midi': self.opens, 'notes': [
            {'index': 1, 'jianzi': '大指五徽勾六弦'},
            {'index': 2, 'jianzi': '挑四徽三分'},
            {'index': 3, 'jianzi': '注下五徽'},
        ]})
        self.assertEqual(warnings, [])

    def test_positioned_pluck_does_not_guess_string_after_open_or_rest(self):
        for predecessor in ['散勾三弦', '泛起五徽挑六弦']:
            context = MODULE.new_context()
            MODULE.parse_jianzi(predecessor, self.opens, context)
            actual, reason = MODULE.parse_jianzi('挑四徽六分', self.opens, context)
            self.assertEqual(actual, [])
            self.assertEqual(reason, 'no_explicit_string')
        context = MODULE.new_context()
        MODULE.parse_jianzi('大指五徽勾六弦', self.opens, context)
        MODULE.advance_context_for_bare_rest(context, {'jianpu': '0（休止）'}, '')
        self.assertEqual(MODULE.parse_jianzi('挑四徽六分', self.opens, context)[0], [])

    def setUp(self):
        self.opens = [48, 50, 53, 55, 57, 60, 62]

    def test_bare_zhuang_cannot_write_a_lower_note_after_pressed_attack(self):
        def warnings(previous_text, previous_pitch, current_text, current_pitch):
            return MODULE.zhuang_direction_warnings({
                'metadata': {'tonic': '1=F'}, 'open_midi': self.opens,
                'notes': [
                    {'index': 74, 'jianpu': previous_pitch, 'jianzi': previous_text},
                    {'index': 75, 'jianpu': current_pitch, 'jianzi': current_text},
                ],
            })

        self.assertEqual(warnings('大指九徽挑二弦', '3̣', '撞', '2̣'), [{
            'index': 75, 'code': 'jianzi_pitch_mismatch',
            'reason': 'zhuang_target_below_previous_attack',
        }])
        self.assertEqual(warnings('大指九徽挑二弦', '3̣', '撞', '4̣'), [])
        self.assertEqual(warnings('大指九徽挑二弦', '3̣', '撞', '3̣'), [])
        self.assertEqual(warnings('大指九徽挑二弦', '2̣', '撞', '1̣'), [])
        self.assertEqual(warnings('大指九徽挑二弦', '3̣', '反撞', '2̣'), [])
        self.assertEqual(warnings('散挑二弦', '3̣', '撞', '2̣'), [])
        self.assertEqual(warnings('', '3̣', '撞', '2̣'), [])

    def test_inherited_gesture_constraints(self):
        def codes(*texts):
            notes = [{'index': i, 'jianpu': '1', 'jianzi': text}
                     for i, text in enumerate(texts)]
            return [(w['index'], w['code']) for w in
                    MODULE.inherited_gesture_warnings({'open_midi': self.opens,
                                                       'notes': notes})]

        open_code = 'open_string_unplayable_gesture'
        self.assertIn((1, open_code), codes('散挑三弦', '注下徽外'))
        self.assertIn((1, open_code), codes('散勾二弦', '猱'))
        self.assertNotIn((1, open_code), codes('大指九徽勾二弦', '注下十徽'))
        self.assertNotIn((2, open_code),
                         codes('大指九徽勾六弦', '散勾五弦', '绰上七徽'))
        self.assertIn((2, open_code),
                      codes('大指九徽勾六弦', '散勾六弦', '绰上七徽'))

        context = MODULE.new_context()
        MODULE.parse_jianzi('大指五徽挑七弦', self.opens, context)
        MODULE.parse_jianzi('散勾五弦', self.opens, context)
        self.assertEqual((context['active_left_string'], context['active_left_hui']),
                         (7, 5.0))
        sounded, reason = MODULE.parse_jianzi('绰上四徽四分', self.opens, context)
        expected, _ = MODULE.position_pitch(7, 4.4, self.opens)
        self.assertIsNone(reason)
        self.assertAlmostEqual(sounded[0], expected)

        context = MODULE.new_context()
        MODULE.parse_jianzi('大指五徽挑七弦', self.opens, context)
        MODULE.parse_jianzi('散勾七弦', self.opens, context)
        self.assertIsNone(context['active_left_string'])
        sounded, reason = MODULE.parse_jianzi('绰上四徽四分', self.opens, context)
        self.assertEqual(sounded, [])
        self.assertEqual(reason, 'context_dependent_slide')
        self.assertIn((1, 'taoqi_without_thumb_origin'),
                      codes('名指九徽勾二弦', '名指十徽滔起'))
        self.assertIn((1, 'taoqi_without_thumb_origin'),
                      codes('名指十徽勾五弦', '大指十徽五弦滔起'))
        self.assertNotIn((1, 'taoqi_without_thumb_origin'),
                         codes('大指九徽挑五弦', '大指十徽五弦滔起'))
        mismatch = MODULE.inherited_gesture_warnings({
            'open_midi': self.opens,
            'notes': [{'index': 0, 'jianzi': '大指九徽挑二弦'},
                      {'index': 1, 'jianzi': '名指十徽五弦滔起'}],
        })
        self.assertTrue(any(w.get('code') == 'taoqi_without_thumb_origin'
                            and w.get('previous_string') == 2 and w.get('string') == 5
                            for w in mismatch))
        self.assertNotIn((2, 'taoqi_without_thumb_origin'),
                         codes('大指九徽勾二弦', '绰大指十徽', '名指十一徽滔起'))
        kneeling_code = 'kneeling_taoqi_below_seventh_hui'
        self.assertNotIn((1, kneeling_code),
                         codes('大指九徽挑七弦', '跪指七徽滔起'))
        self.assertNotIn((1, kneeling_code),
                         codes('大指九徽挑六弦', '跪指五徽六分滔起'))
        self.assertIn((1, kneeling_code),
                      codes('大指九徽挑六弦', '跪指七徽一分滔起'))
        self.assertIn((1, kneeling_code),
                      codes('大指九徽挑六弦', '跪指八徽滔起'))
        self.assertIn((1, kneeling_code),
                      codes('大指九徽挑六弦', '跪指滔起'))
        self.assertNotIn((1, kneeling_code),
                         codes('大指六徽挑六弦', '跪指滔起'))
        self.assertIn((3, 'repeated_fu_without_walk'),
                      codes('大指九徽挑三弦', '退', '复', '复'))
        self.assertNotIn((3, 'repeated_fu_without_walk'),
                         codes('大指九徽挑三弦', '退', '复', '绰上八徽'))

        harmonic_code = 'harmonic_ornament_without_stopped'
        self.assertIn((1, harmonic_code),
                      codes('泛起名指十徽挑六弦', '名指十徽勾六弦吟'))
        self.assertIn((1, harmonic_code),
                      codes('泛起名指十徽挑六弦', '猱'))
        self.assertIn((0, harmonic_code),
                      codes('泛起名指十徽挑六弦吟'))
        self.assertNotIn((1, harmonic_code),
                         codes('泛起名指十徽挑六弦', '按音名指十徽勾六弦吟'))
        self.assertNotIn((2, harmonic_code),
                         codes('泛起名指十徽挑六弦', '按音名指十徽勾六弦', '吟'))

    def test_comma_compound_replays_pitch_without_format_warning(self):
        context = MODULE.new_context()
        text = '撮（大指九徽三弦、六弦散音）'
        actual, reason = MODULE.parse_jianzi(text, self.opens, context)
        expected, _ = MODULE.position_pitch(3, 9.0, self.opens)
        self.assertIsNone(reason)
        self.assertEqual(actual, [expected, self.opens[5]])
        self.assertEqual(context['left_positions']['大指'],
                         {'string': 3, 'hui': 9.0})
        notes = [{'index': 0, 'jianzi': text},
                 {'index': 1, 'jianzi': '名指三弦十徽滔起'}]
        warnings = MODULE.inherited_gesture_warnings(
            {'open_midi': self.opens, 'notes': notes})
        self.assertFalse(any(w['index'] == 0 for w in warnings))
        self.assertNotIn((1, 'taoqi_without_thumb_origin'),
                         [(w['index'], w['code']) for w in warnings])

    def test_taoqi_uses_still_held_thumb_not_fixed_note_lookback(self):
        def warning_codes(*texts):
            return [(w['index'], w['code']) for w in
                    MODULE.inherited_gesture_warnings({
                        'open_midi': self.opens,
                        'notes': [{'index': i, 'jianzi': text}
                                  for i, text in enumerate(texts)],
                    })]

        self.assertNotIn((2, 'taoqi_without_thumb_origin'), warning_codes(
            '大指五徽挑七弦猱', '中指五徽三分挑五弦',
            '名指七徽七弦滔起'))
        self.assertIn((2, 'taoqi_without_thumb_origin'), warning_codes(
            '大指五徽挑七弦', '中指五徽三分挑七弦',
            '名指七徽七弦滔起'))

    def test_bare_rest_keeps_open_abbreviation_separate_from_held_position(self):
        context = MODULE.new_context()
        for text in ('中指七徽勾五弦', '散勾三弦'):
            MODULE.parse_jianzi(text, self.opens, context)
        rest = {'index': 2, 'jianpu': '0（休止）', 'abc': 'z'}
        self.assertTrue(MODULE.advance_context_for_bare_rest(context, rest, ''))
        self.assertIsNone(context['sound_mode'])
        self.assertEqual((context['active_left_string'], context['active_left_hui']),
                         (5, 7.0))
        sounded, reason = MODULE.parse_jianzi('勾四弦', self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(sounded, [self.opens[3]])
        self.assertEqual(context['current_note_attacks'][0]['mode'], 'open')

        # Consecutive rests must carry the same default; an explicit pressed
        # attack afterward switches subsequent shorthand back to stopped.
        context = MODULE.new_context()
        for text in ('中指七徽勾五弦', '散勾三弦'):
            MODULE.parse_jianzi(text, self.opens, context)
        for index in (2, 3):
            self.assertTrue(MODULE.advance_context_for_bare_rest(
                context, {'index': index, 'jianpu': '0（休止）', 'abc': 'z'}, ''))
        sounded, reason = MODULE.parse_jianzi('绰上六徽', self.opens, context)
        expected, _ = MODULE.position_pitch(5, 6.0, self.opens)
        self.assertIsNone(reason)
        self.assertAlmostEqual(sounded[0], expected)
        MODULE.parse_jianzi('中指七徽勾五弦', self.opens, context)
        sounded, reason = MODULE.parse_jianzi('勾五弦', self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(context['current_note_attacks'][0]['mode'], 'stopped')
        self.assertEqual(sounded, [69.0])

    def test_suffix_walk_is_held_through_other_string_open_yinghe(self):
        context = MODULE.new_context()
        attack, reason = MODULE.parse_jianzi(
            '名指十徽勾2弦绰上九徽', self.opens, context)
        ten_hui, _ = MODULE.position_pitch(2, 10.0, self.opens)
        self.assertIsNone(reason)
        self.assertEqual(attack, [ten_hui])
        self.assertEqual((context['active_left_string'], context['active_left_hui']),
                         (2, 9.0))
        MODULE.parse_jianzi('散挑5弦应合', self.opens, context)
        self.assertEqual((context['active_left_string'], context['active_left_hui']),
                         (2, 9.0))
        notes = [
            {'index': 230, 'jianzi': '名指十徽勾2弦绰上九徽'},
            {'index': 231, 'jianpu': '|', 'jianzi': ''},
            {'index': 232, 'jianzi': '散挑5弦应合'},
            {'index': 233, 'jianzi': '注下十徽'},
        ]
        self.assertFalse(MODULE.walk_motion_warnings(
            {'open_midi': self.opens, 'notes': notes}))

    def test_single_stopped_note(self):
        pitches, reason = MODULE.parse_jianzi("名指九徽勾5弦", self.opens)
        self.assertIsNone(reason)
        self.assertAlmostEqual(pitches[0], 64.01955, places=4)

    def test_invalid_string_numbers_are_reported_and_not_pitch_matched(self):
        for text, numbers in [
            ('按音大指四徽挑九弦', [9]), ('散挑8弦', [8]),
            ('泛音名指十一徽勾十一弦', [11]), ('挑0弦', [0]),
            ('撮（大指七徽六弦泛音＋中指七徽九弦泛音）', [9]),
            ('泛音食指五徽拂一至八弦', [8]),
        ]:
            with self.subTest(text=text):
                self.assertEqual(MODULE.invalid_string_numbers(text), numbers)
                self.assertEqual(MODULE.parse_jianzi(text, self.opens), ([], 'invalid_string_number'))
                self.assertEqual(MODULE.invalid_string_warnings({'notes': [
                    {'index': 82, 'jianzi': text}]}), [
                        {'index': 82, 'code': 'invalid_string_number', 'strings': numbers}])

    def test_valid_compact_strings_and_hui_are_not_invalid_strings(self):
        for text in ['名指九徽勾四弦', '食指十三徽勾七弦',
                     '散历六五弦', '泛音名指五徽拂一至六弦',
                     '剔五弦散如一四弦', '撮四七弦', '撮47弦', '历7六弦', '历3二弦']:
            with self.subTest(text=text):
                self.assertEqual(MODULE.invalid_string_numbers(text), [])

    def test_explicit_double_harmonics(self):
        for technique in ('撮', '泼', '剌'):
            for text in (f'{technique}（大指七徽六弦泛音＋中指七徽一弦泛音）',
                         f'{technique}(泛音大指七徽六弦+泛音中指七徽一弦)'):
                with self.subTest(text=text):
                    pitches, reason = MODULE.parse_jianzi(text, self.opens)
                    self.assertIsNone(reason)
                    self.assertEqual(pitches, [72, 60])

    def test_double_harmonic_chord_audit_matches_and_detects_wrong_pitch(self):
        text = '撮（大指七徽六弦泛音＋中指七徽一弦泛音）'
        data = {'metadata': {'tonic': '1=C'}, 'open_midi': self.opens,
                'notes': [{'index': 0, 'jianpu': '1', 'jianpu_alt': '1̇', 'jianzi': text}]}
        self.assertEqual(MODULE.audit(data, 50)['details'][0]['status'], 'matched')
        data['notes'][0]['jianpu_alt'] = '2̇'
        self.assertEqual(MODULE.audit(data, 50)['details'][0]['status'], 'mismatched')

    def test_double_harmonic_context_is_not_a_stopped_position(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi('大指七徽勾一弦', self.opens, context)
        MODULE.parse_jianzi('撮（大指七徽六弦泛音＋中指七徽一弦泛音）', self.opens, context)
        self.assertIsNone(context['active_left_hui'])
        self.assertFalse(context['harmonic_scope'])
        context['harmonic_scope'] = True
        pitches, reason = MODULE.parse_jianzi('勾二弦', self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [62])

    def test_double_harmonic_stop_and_different_hui(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi('泛起撮（大指七徽六弦泛音＋中指七徽一弦泛音）泛止', self.opens, context)
        self.assertFalse(context['harmonic_scope'])
        self.assertIsNone(context['harmonic_hui'])
        pitches, reason = MODULE.parse_jianzi('撮（大指七徽六弦泛音＋中指五徽一弦泛音）', self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [72, 67])
        self.assertIsNone(context['harmonic_hui'])

    def test_double_harmonic_incomplete_positions_are_not_guessed(self):
        pitches, reason = MODULE.parse_jianzi('撮（大指七徽六弦泛音＋中指一弦泛音）', self.opens)
        self.assertEqual(pitches, [])
        self.assertEqual(reason, 'double_harmonic_position_missing')

    def test_fu_restores_walk_origin_and_allows_repeated_slide(self):
        context = MODULE.new_context()
        origin, _ = MODULE.parse_jianzi("大指七徽九分挑四弦", self.opens, context)
        MODULE.parse_jianzi("绰上七徽", self.opens, context)
        restored, reason = MODULE.parse_jianzi("复", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(origin, restored)
        self.assertEqual(context["active_left_hui"], 7.9)
        notes = [{"index": i, "jianzi": text} for i, text in enumerate([
            "大指七徽九分挑四弦", "绰上七徽", "复", "绰上七徽"])]
        self.assertEqual(MODULE.walk_motion_warnings({"open_midi": self.opens, "notes": notes}), [])

    def test_bare_movement_invalidates_unknown_hui_but_fu_restores_origin(self):
        for movement in ("进", "退", "急进", "急退"):
            with self.subTest(movement=movement):
                context = MODULE.new_context()
                MODULE.parse_jianzi("大指七徽挑四弦", self.opens, context)
                pitches, reason = MODULE.parse_jianzi(movement, self.opens, context)
                self.assertFalse(pitches)
                self.assertIsNotNone(reason)
                self.assertIsNone(context["active_left_hui"])
                MODULE.parse_jianzi("复", self.opens, context)
                self.assertEqual(context["active_left_hui"], 7.0)
                notes = [{"index": i, "jianzi": t} for i, t in enumerate([
                    "大指七徽挑四弦", movement, "绰上七徽"])]
                self.assertEqual(MODULE.walk_motion_warnings({"open_midi": self.opens, "notes": notes}), [])

    def test_positioned_jin_tui_update_hui(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("大指七徽挑四弦", self.opens, context)
        MODULE.parse_jianzi("进六徽二分", self.opens, context)
        self.assertEqual(context["active_left_hui"], 6.2)
        MODULE.parse_jianzi("退七徽六分", self.opens, context)
        self.assertEqual(context["active_left_hui"], 7.6)

    def test_new_attack_and_rest_clear_fu_origin(self):
        for reset in ("大指七徽挑四弦", None):
            context = MODULE.new_context()
            MODULE.parse_jianzi("大指七徽九分挑四弦", self.opens, context)
            MODULE.parse_jianzi("绰上七徽", self.opens, context)
            if reset is None:
                MODULE.advance_context_for_bare_rest(context, {"jianpu": "0（休止）", "jianzi": ""})
            else:
                MODULE.parse_jianzi(reset, self.opens, context)
            self.assertIsNone(context["walk_return_position"])
            pitches, reason = MODULE.parse_jianzi("复", self.opens, context)
            self.assertEqual(pitches, [])
            self.assertEqual(reason, "context_dependent_slide")

    def test_hui_outside_positions_have_project_defined_pitches(self):
        base, _ = MODULE.position_pitch(5, 12.3, self.opens)
        outer, reason = MODULE.position_pitch(5, "徽外", self.opens)
        outer_half, half_reason = MODULE.position_pitch(5, "徽外半", self.opens)
        self.assertIsNone(reason)
        self.assertIsNone(half_reason)
        self.assertAlmostEqual(outer, base - 1.0)
        self.assertAlmostEqual(outer_half, base - 2.0)
        pitches, reason = MODULE.parse_jianzi("名指徽外半勾5弦", self.opens)
        self.assertIsNone(reason)
        self.assertAlmostEqual(pitches[0], outer_half)

    def test_compound_has_two_unordered_pitches(self):
        text = "撮（大指七徽九分6弦按音＋3弦散音）"
        pitches, reason = MODULE.parse_jianzi(text, self.opens)
        self.assertIsNone(reason)
        self.assertEqual(len(pitches), 2)
        pairs = MODULE.best_pairing([53, 69], pitches)
        self.assertEqual([p["jianpu_name"] for p in pairs], ["F3", "A4"])
        self.assertTrue(all(p["absolute_cents"] < 50 for p in pairs))

    def test_compound_accepts_open_then_stopped_order(self):
        text = "撮（六弦散音＋七徽一弦按音）"
        pitches, reason = MODULE.parse_jianzi(text, self.opens)
        self.assertIsNone(reason)
        self.assertEqual(len(pitches), 2)

    def test_compound_accepts_implicit_stopped_component(self):
        text = "撮（六弦散音＋中指七徽一弦）"
        pitches, reason = MODULE.parse_jianzi(text, self.opens)
        self.assertIsNone(reason)
        self.assertEqual(len(pitches), 2)

    def test_compound_accepts_string_then_hui_component(self):
        text = "撮（六弦散音＋一弦七徽按音）"
        pitches, reason = MODULE.parse_jianzi(text, self.opens)
        self.assertIsNone(reason)
        self.assertEqual(len(pitches), 2)

    def test_compound_accepts_chinese_string_numbers_and_keeps_stopped_state(self):
        context = MODULE.new_context()
        pitches, reason = MODULE.parse_jianzi(
            "撮（大指七徽五弦按音＋三弦散音）", self.opens, context
        )
        self.assertIsNone(reason)
        self.assertEqual(len(pitches), 2)
        self.assertEqual(context["stopped_string"], 5)
        self.assertEqual(context["active_left_string"], 5)
        self.assertAlmostEqual(context["stopped_hui"], 7.0)

    def test_harmonic_compound_updates_hui_for_following_abbreviated_pluck(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("泛起名指五徽挑6弦", self.opens, context)
        MODULE.parse_jianzi(
            "撮（食指四徽3弦按音＋中指四徽1弦按音）",
            self.opens,
            context,
        )

        pitches, reason = MODULE.parse_jianzi("挑2弦", self.opens, context)
        expected, error = MODULE.position_pitch(2, 4.0, self.opens, "harmonic")
        self.assertIsNone(reason)
        self.assertIsNone(error)
        self.assertEqual(context["harmonic_hui"], 4.0)
        self.assertEqual(pitches, [expected])

    def test_slide_without_inherited_string_is_skipped(self):
        pitches, reason = MODULE.parse_jianzi(
            "注下七徽九分", self.opens, MODULE.new_context()
        )
        self.assertEqual(pitches, [])
        self.assertEqual(reason, "context_dependent_slide")

    def test_positioned_taoqi_updates_inherited_position_for_following_plucks(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("大指四徽六分挑7弦", self.opens, context)

        taoqi, reason = MODULE.parse_jianzi("跪指五徽滔起", self.opens, context)
        expected_taoqi, _ = MODULE.position_pitch(7, 5.0, self.opens, "stopped")
        self.assertIsNone(reason)
        self.assertAlmostEqual(taoqi[0], expected_taoqi)
        self.assertEqual(context["active_left_string"], 7)
        self.assertEqual(context["active_left_hui"], 5.0)
        self.assertEqual(context["left_finger"], "跪指")

        inherited_sixth, reason = MODULE.parse_jianzi("就勾6弦", self.opens, context)
        expected_sixth, _ = MODULE.position_pitch(6, 5.0, self.opens, "stopped")
        self.assertIsNone(reason)
        self.assertAlmostEqual(inherited_sixth[0], expected_sixth)

        inherited_seventh, reason = MODULE.parse_jianzi("挑7弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertAlmostEqual(inherited_seventh[0], expected_taoqi)

    def test_zhi_string_transition_is_skipped(self):
        pitches, reason = MODULE.parse_jianzi("至5弦", self.opens)
        self.assertEqual(pitches, [])
        self.assertEqual(reason, "context_dependent_transition")

    def test_open_string(self):
        pitches, reason = MODULE.parse_jianzi("勾4弦", self.opens)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [55])

    def test_open_string_accepts_chinese_string_number(self):
        pitches, reason = MODULE.parse_jianzi("散托七弦", self.opens)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [62])

    def test_compact_chinese_string_numbers_share_trailing_xian(self):
        self.assertEqual(MODULE.parse_string_numbers("散撮四七弦"), [4, 7])
        pitches, reason = MODULE.parse_jianzi("散撮四七弦", self.opens)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [55, 62])

    def test_ruyi_is_not_parsed_as_string_one(self):
        self.assertEqual(MODULE.parse_string_numbers("剔五弦散如一四弦"), [5, 4])
        context = MODULE.new_context()
        MODULE.parse_jianzi("名指七徽六分勾五弦", self.opens, context)
        pitches, reason = MODULE.parse_jianzi(
            "剔五弦散如一四弦", self.opens, context
        )
        self.assertIsNone(reason)
        self.assertAlmostEqual(pitches[0], 67.038015, places=5)
        self.assertEqual(pitches[1], 55)
        self.assertEqual(context["active_left_string"], 5)
        self.assertEqual(context["active_left_hui"], 7.6)

    def test_compact_double_open_string_chord_audits_both_written_pitches(self):
        report = MODULE.audit({
            "metadata": {"tonic": "1=G"},
            "open_midi": self.opens,
            "notes": [{
                "index": 407,
                "jianpu": "1̣",
                "jianpu_alt": "5̣",
                "jianzi": "散撮四七弦",
            }],
        }, 50.0)
        self.assertEqual(report["details"][0]["status"], "matched")
        self.assertEqual(
            [pair["absolute_cents"] for pair in report["details"][0]["pairs"]],
            [0.0, 0.0],
        )

    def test_explicit_stopped_note_overrides_harmonic_scope_for_one_event(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("泛起食指七徽勾一弦", self.opens, context)
        stopped, reason = MODULE.parse_jianzi(
            "按音大指九徽勾七弦", self.opens, context
        )
        expected, _ = MODULE.position_pitch(7, 9.0, self.opens, "stopped")
        self.assertIsNone(reason)
        self.assertAlmostEqual(stopped[0], expected)
        self.assertTrue(context["harmonic_scope"])
        # The following unmarked note returns to inherited harmonic mode;
        # only 泛止 ends the region.
        harmonic, reason = MODULE.parse_jianzi("大指七徽挑五弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(harmonic, [69])

    def test_standalone_fanqi_opens_harmonic_scope(self):
        context = MODULE.new_context()
        pitches, reason = MODULE.parse_jianzi("泛起", self.opens, context)
        self.assertEqual(pitches, [])
        self.assertEqual(reason, "ornament_or_control")
        self.assertTrue(context["harmonic_scope"])
        harmonic, reason = MODULE.parse_jianzi(
            "名指九徽挑四弦", self.opens, context
        )
        expected, _ = MODULE.position_pitch(4, 9.0, self.opens, "harmonic")
        self.assertIsNone(reason)
        self.assertAlmostEqual(harmonic[0], expected)

    def test_explicit_fanyin_marks_current_note_harmonic(self):
        pitches, reason = MODULE.parse_jianzi("泛音名指九徽挑四弦", self.opens)
        expected, _ = MODULE.position_pitch(4, 9.0, self.opens, "harmonic")
        self.assertIsNone(reason)
        self.assertAlmostEqual(pitches[0], expected)

    def test_harmonic_region_rejects_standalone_walk(self):
        warnings = MODULE.harmonic_continuity_warnings({
            "open_midi": self.opens,
            "notes": [
                {"index": 1, "jianpu": "5", "jianzi": "泛起中指七徽勾一弦"},
                {"index": 2, "jianpu": "6", "jianzi": "绰上六徽"},
            ],
        })
        self.assertEqual(warnings, [{
            "index": 2,
            "code": "walk_inside_harmonic_region",
            "technique": "绰上",
        }])

    def test_fractional_harmonic_position_reports_warning(self):
        for glyph in ['大指二徽三分挑五弦', '泛音大指二徽三分挑五弦',
                      '撮（大指二徽三分五弦泛音＋中指七徽一弦泛音）']:
            warnings = MODULE.harmonic_continuity_warnings({
                'open_midi': self.opens, 'notes': [
                    {'index': 0, 'jianpu': '2', 'jianzi': '泛起大指七徽挑五弦'},
                    {'index': 1, 'jianpu': '3', 'jianzi': glyph},
                ],
            })
            self.assertIn('fractional_hui_in_harmonic', [w['code'] for w in warnings], glyph)

    def test_fractional_stopped_and_integer_harmonic_are_allowed(self):
        for glyph in ['泛音大指二徽挑五弦', '大指二徽三分挑五弦按音',
                      '泛止大指二徽三分挑五弦']:
            warnings = MODULE.harmonic_continuity_warnings({
                'open_midi': self.opens, 'notes': [
                    {'index': 0, 'jianpu': '2', 'jianzi': '泛起大指七徽挑五弦'},
                    {'index': 1, 'jianpu': '3', 'jianzi': glyph},
                ],
            })
            self.assertNotIn('fractional_hui_in_harmonic', [w['code'] for w in warnings], glyph)

    def test_harmonic_same_string_reattack_warns(self):
        warnings = MODULE.harmonic_continuity_warnings({
            "open_midi": self.opens,
            "notes": [
                {"index": 71, "jianpu": "5", "jianzi": "泛音中指七徽勾一弦"},
                {"index": 72, "jianpu": "3", "jianzi": "泛音大指四徽挑一弦"},
            ],
        })
        self.assertEqual(warnings, [{
            "index": 72,
            "code": "harmonic_to_same_string_attack",
            "strings": [1],
        }])

    def test_written_stop_clears_open_resonance_across_barline(self):
        def warnings(first):
            return MODULE.open_to_stopped_transition_warnings({
                "open_midi": self.opens,
                "notes": [
                    {"index": 1, "jianpu": "2", "jianzi": first},
                    {"index": 2, "jianpu": "|", "jianzi": ""},
                    {"index": 3, "jianpu": "3", "jianzi": "名指十徽挑七弦"},
                ],
            })
        self.assertEqual(warnings("散挑七弦停"), [])
        self.assertEqual([w["index"] for w in warnings("散挑七弦")], [3])

    def test_standalone_stop_clears_open_and_harmonic_resonance(self):
        open_notes = [
            {"index": 1, "jianpu": "2", "jianzi": "散挑七弦"},
            {"index": 2, "jianpu": "－（延音）", "jianzi": "停"},
            {"index": 3, "jianpu": "3", "jianzi": "名指十徽挑七弦"},
        ]
        self.assertEqual(MODULE.open_to_stopped_transition_warnings({
            "open_midi": self.opens, "notes": open_notes,
        }), [])
        harmonic_notes = [
            {"index": 1, "jianpu": "2", "jianzi": "泛起食指三徽勾四弦"},
            {"index": 2, "jianpu": "－（延音）", "jianzi": "停泛止"},
            {"index": 3, "jianpu": "3", "jianzi": "大指五徽挑四弦"},
        ]
        self.assertNotIn("harmonic_to_same_string_attack", [w["code"] for w in
            MODULE.harmonic_continuity_warnings({
                "open_midi": self.opens, "notes": harmonic_notes,
            })])

    def test_written_stop_clears_harmonic_resonance_across_barline(self):
        def warnings(first):
            return MODULE.harmonic_continuity_warnings({
                "open_midi": self.opens,
                "notes": [
                    {"index": 1, "jianpu": "2", "jianzi": first},
                    {"index": 2, "jianpu": "|", "jianzi": ""},
                    {"index": 3, "jianpu": "3", "jianzi": "大指五徽挑四弦"},
                ],
            })
        self.assertEqual(warnings("泛起食指三徽勾四弦停"), [])
        self.assertIn("harmonic_to_same_string_attack",
                      [w["code"] for w in warnings("泛起食指三徽勾四弦")])

    def test_open_to_stopped_requires_stopped_attack_not_harmonic(self):
        for first in ("散挑七弦", "散挑七弦停"):
            data = {"open_midi": self.opens, "notes": [
                {"index": 285, "jianpu": "3", "jianzi": first},
                {"index": 286, "jianpu": "|", "jianzi": ""},
                {"index": 287, "jianpu": "3̇̇", "jianzi": "泛起名指十徽历七弦"},
            ]}
            self.assertEqual(MODULE.open_to_stopped_transition_warnings(data), [])

    def test_open_to_stopped_exempts_yan_but_keeps_its_stopped_pitch(self):
        data = {"open_midi": self.opens, "notes": [
            {"index": 1, "jianpu": "2", "jianzi": "散勾六弦"},
            {"index": 2, "jianpu": "3", "jianzi": "名指徽外罨六弦"},
        ]}
        self.assertEqual(MODULE.open_to_stopped_transition_warnings(data), [])
        context = MODULE.new_context()
        MODULE.parse_jianzi("散勾六弦", self.opens, context)
        actual, reason = MODULE.parse_jianzi("名指徽外罨六弦", self.opens, context)
        expected, error = MODULE.position_pitch(6, "徽外", self.opens, "stopped")
        self.assertIsNone(reason)
        self.assertIsNone(error)
        self.assertEqual(actual, [expected])
        self.assertEqual(context["current_note_attacks"][0]["mode"], "stopped")

        data["notes"][1]["jianzi"] = "名指十徽挑六弦"
        self.assertEqual([w["code"] for w in
                          MODULE.open_to_stopped_transition_warnings(data)],
                         ["open_to_stopped_same_string"])

    def test_same_harmonic_position_alternating_right_hand_is_exempt(self):
        for second in ['食指七徽挑二弦', '挑二弦']:
            warnings = MODULE.harmonic_continuity_warnings({'open_midi': self.opens,
                'notes': [{'index': 0, 'jianpu': '2', 'jianzi': '泛起食指七徽勾二弦'},
                          {'index': 1, 'jianpu': '2', 'jianzi': second}]})
            self.assertEqual(warnings, [])

    def test_same_pitch_different_harmonic_hui_has_specific_advisory(self):
        warnings = MODULE.harmonic_continuity_warnings({
            'open_midi': self.opens,
            'notes': [
                {'index': 0, 'jianpu': '2̇', 'jianzi': '泛起食指三徽勾四弦'},
                {'index': 1, 'jianpu': '2̇', 'jianzi': '大指六徽挑四弦'},
            ],
        })
        self.assertEqual(len(warnings), 1)
        self.assertTrue(warnings[0]['same_pitch_different_hui'])

    def test_suffix_fanzhi_same_hui_continuity_is_exempt(self):
        for first, second in [
            ('泛音食指三徽勾四弦', '泛音食指三徽挑四弦泛止'),
            ('泛起食指三徽勾四弦', '挑四弦泛止'),
        ]:
            with self.subTest(second=second):
                data = {'open_midi': self.opens, 'notes': [
                    {'index': 90, 'jianpu': '2̇', 'jianzi': first},
                    {'index': 91, 'jianpu': '2̇', 'jianzi': second},
                ]}
                self.assertEqual(MODULE.harmonic_continuity_warnings(data), [])

    def test_suffix_fanzhi_different_hui_keeps_specific_advisory(self):
        data = {'open_midi': self.opens, 'notes': [
            {'index': 90, 'jianpu': '2̇', 'jianzi': '泛音食指八徽勾四弦'},
            {'index': 91, 'jianpu': '2̇', 'jianzi': '泛音食指三徽挑四弦泛止'},
        ]}
        warnings = MODULE.harmonic_continuity_warnings(data)
        self.assertEqual(len(warnings), 1)
        self.assertTrue(warnings[0]['same_pitch_different_hui'])

    def test_parser_keeps_current_attack_evidence_after_suffix_stop(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi('泛起食指三徽勾四弦', self.opens, context)
        MODULE.parse_jianzi('挑四弦泛止', self.opens, context)
        self.assertEqual(context['current_note_attacks'], [
            {'string': 4, 'hui': 3.0, 'mode': 'harmonic'},
        ])
        self.assertFalse(context['harmonic_scope'])
        MODULE.parse_jianzi('泛止食指三徽挑四弦', self.opens, context)
        self.assertEqual(context['current_note_attacks'][0]['mode'], 'stopped')

    def test_harmonic_wheel_inherits_same_hui_and_is_exempt(self):
        for wheel in ['轮四弦', '輪四弦']:
            data = {'open_midi': self.opens, 'notes': [
                {'index': 0, 'jianpu': '2̇', 'jianzi': '泛起大指十一徽挑四弦'},
                {'index': 1, 'jianpu': '2̇', 'jianzi': wheel},
            ]}
            self.assertEqual(MODULE.harmonic_continuity_warnings(data), [])
            context = MODULE.new_context()
            first, _ = MODULE.parse_jianzi(data['notes'][0]['jianzi'], self.opens, context)
            second, reason = MODULE.parse_jianzi(wheel, self.opens, context)
            self.assertIsNone(reason)
            self.assertEqual(first, second)

    def test_harmonic_region_warns_on_other_left_hand_walks(self):
        for glyph in ['上三徽二分', '下九徽', '进五徽', '退八徽', '淌七徽', '浒上七徽', '引上七徽']:
            warnings = MODULE.harmonic_continuity_warnings({
                'open_midi': self.opens, 'notes': [
                    {'index': 0, 'jianpu': '2', 'jianzi': '泛起大指十一徽挑四弦'},
                    {'index': 1, 'jianpu': '3', 'jianzi': glyph},
                ],
            })
            self.assertIn('walk_inside_harmonic_region', [w['code'] for w in warnings], glyph)

    def test_walk_after_fanzhi_is_not_harmonic_walk(self):
        warnings = MODULE.harmonic_continuity_warnings({
            'open_midi': self.opens, 'notes': [
                {'index': 0, 'jianpu': '2', 'jianzi': '泛起大指十一徽挑四弦'},
                {'index': 1, 'jianpu': '3', 'jianzi': '泛止上三徽二分'},
            ],
        })
        self.assertNotIn('walk_inside_harmonic_region', [w['code'] for w in warnings])

    def test_different_pitch_or_stopped_note_keeps_general_advisory(self):
        for target, glyph in [
            ('2', '大指六徽挑四弦'),
            ('2̇', '大指六徽挑四弦按音'),
        ]:
            warnings = MODULE.harmonic_continuity_warnings({
                'open_midi': self.opens,
                'notes': [
                    {'index': 0, 'jianpu': '2̇', 'jianzi': '泛起食指三徽勾四弦'},
                    {'index': 1, 'jianpu': target, 'jianzi': glyph},
                ],
            })
            self.assertEqual(len(warnings), 1)
            self.assertNotIn('same_pitch_different_hui', warnings[0])

    def test_harmonic_same_position_is_exempt_without_technique_enumeration(self):
        for second in ['食指七徽勾二弦', '食指七徽二弦', '二弦',
                       '勾剔二弦', '抹勾二弦', '涓二弦', '自定义指法二弦']:
            warnings = MODULE.harmonic_continuity_warnings({'open_midi': self.opens,
                'notes': [{'index': 0, 'jianpu': '2', 'jianzi': '泛起食指七徽勾二弦'},
                          {'index': 1, 'jianpu': '2', 'jianzi': second}]})
            self.assertEqual(warnings, [], second)

    def test_stopped_reattack_is_not_exempt_by_alternating_right_hand(self):
        warnings = MODULE.harmonic_continuity_warnings({'open_midi': self.opens,
            'notes': [{'index': 0, 'jianpu': '2', 'jianzi': '泛起食指七徽勾二弦'},
                      {'index': 1, 'jianpu': '2', 'jianzi': '食指七徽挑二弦按音'}]})
        self.assertEqual([w['code'] for w in warnings], ['harmonic_to_same_string_attack'])

    def test_li_resonance_uses_first_and_last_attack_in_order(self):
        cases = [
            (['泛起五徽挑五弦', '大指九徽历六五弦'], []),
            (['泛起五徽挑六弦', '大指九徽历六五弦'], [6]),
            (['泛起五徽历六五弦', '九徽挑六弦'], []),
            (['泛起五徽历六五弦', '九徽挑五弦'], [5]),
        ]
        for texts, expected in cases:
            with self.subTest(texts=texts):
                warnings = MODULE.harmonic_continuity_warnings({
                    'open_midi': self.opens,
                    'notes': [{'index': i, 'jianpu': '1', 'jianzi': t}
                              for i, t in enumerate(texts)],
                })
                self.assertEqual([s for w in warnings
                                  if w['code'] == 'harmonic_to_same_string_attack'
                                  for s in w['strings']], expected)

    def test_shaoxi_ends_resonance_but_preserves_harmonic_pitch_state(self):
        notes = [{'index': 1, 'jianpu': '1', 'jianzi': '泛起五徽挑六弦少息'},
                 {'index': 2, 'jianpu': '1', 'jianzi': '大指十三徽历六五弦'}]
        self.assertEqual(MODULE.harmonic_continuity_warnings({
            'open_midi': self.opens, 'notes': notes}), [])
        context = MODULE.new_context()
        MODULE.parse_jianzi(notes[0]['jianzi'], self.opens, context)
        MODULE.parse_jianzi(notes[1]['jianzi'], self.opens, context)
        self.assertTrue(context['harmonic_scope'])
        self.assertTrue(all(a['mode'] == 'harmonic' for a in context['current_note_attacks']))

    def test_bare_rest_ends_harmonic_scope_for_continuity_warning(self):
        warnings = MODULE.harmonic_continuity_warnings({
            "open_midi": self.opens,
            "notes": [
                {"index": 1, "jianpu": "5", "jianzi": "泛起中指七徽勾一弦"},
                {"index": 2, "jianpu": "0（休止）", "jianzi": ""},
                {"index": 3, "jianpu": "3", "jianzi": "名指九徽挑一弦"},
            ],
        })
        self.assertEqual(warnings, [])

    def test_swept_fu_skips_single_pitch_comparison(self):
        report = MODULE.audit({
            "open_midi": self.opens,
            "notes": [{
                "index": 1, "jianpu": "5",
                "jianzi": "泛音名指五徽拂一至六弦",
            }],
        }, 50.0)
        self.assertEqual(report["details"][0]["status"], "skipped")
        self.assertEqual(
            report["details"][0]["reason"], "compound_sweep_pitch_ambiguous"
        )

    def test_fu_predecessor_does_not_carry_open_string_warning(self):
        warnings = MODULE.open_to_stopped_transition_warnings({
            "open_midi": self.opens,
            "notes": [
                {"index": 1, "jianpu": "5", "jianzi": "散勾一弦拂"},
                {"index": 2, "jianpu": "5", "jianzi": "大指七徽挑一弦"},
            ],
        })
        self.assertEqual(warnings, [])

    def test_lock_predecessor_does_not_carry_harmonic_warning(self):
        warnings = MODULE.harmonic_continuity_warnings({
            "open_midi": self.opens,
            "notes": [
                {"index": 1, "jianpu": "5", "jianzi": "泛音中指七徽勾一弦锁"},
                {"index": 2, "jianpu": "3", "jianzi": "泛音大指四徽挑一弦"},
            ],
        })
        self.assertEqual(warnings, [])

    def test_four_standalone_walks_without_attack_warns(self):
        warnings = MODULE.consecutive_walk_warnings({
            "open_midi": self.opens,
            "notes": [
                {"index": 1, "jianpu": "5", "jianzi": "大指九徽勾四弦"},
                {"index": 2, "jianpu": "6", "jianzi": "绰上八徽"},
                {"index": 3, "jianpu": "7", "jianzi": "绰上七徽"},
                {"index": 4, "jianpu": "1̇", "jianzi": "绰上六徽"},
                {"index": 5, "jianpu": "2̇", "jianzi": "绰上五徽"},
            ],
        })
        self.assertEqual(warnings, [{
            "index": 5,
            "code": "consecutive_walks_without_attack",
            "count": 4,
        }])

    def test_wanglai_allows_five_events_including_blanks_then_warns(self):
        notes = [
            {"index": 1, "jianpu": "1", "jianzi": "名指七徽勾三弦"},
            {"index": 2, "jianpu": "2", "jianzi": "往来"},
            {"index": 3, "jianpu": "1", "jianzi": ""},
            {"index": 4, "jianpu": "2", "jianzi": ""},
            {"index": 5, "jianpu": "1", "jianzi": ""},
            {"index": 6, "jianpu": "6̣", "jianzi": "注下七徽九分"},
        ]
        self.assertEqual(MODULE.consecutive_walk_warnings({
            "open_midi": self.opens, "notes": notes,
        }), [])
        notes.append({"index": 7, "jianpu": "5̣", "jianzi": ""})
        self.assertEqual(MODULE.consecutive_walk_warnings({
            "open_midi": self.opens, "notes": notes,
        }), [{"index": 7, "code": "consecutive_walks_without_attack", "count": 6}])

    def test_fu_and_blanks_count_but_new_attack_resets_walk_run(self):
        notes = [
            {"index": 1, "jianpu": "1", "jianzi": "名指七徽勾三弦"},
            {"index": 2, "jianpu": "2", "jianzi": "绰上六徽"},
            {"index": 3, "jianpu": "1", "jianzi": ""},
            {"index": 4, "jianpu": "2", "jianzi": "复"},
            {"index": 5, "jianpu": "1", "jianzi": "注下七徽"},
            {"index": 6, "jianpu": "1", "jianzi": "滔起"},
            {"index": 7, "jianpu": "2", "jianzi": "绰上六徽"},
        ]
        self.assertEqual(MODULE.consecutive_walk_warnings({
            "open_midi": self.opens, "notes": notes,
        }), [{"index": 5, "code": "consecutive_walks_without_attack", "count": 4}])

    def test_prefix_fanzhi_ends_harmonic_before_current_note_and_preserves_position(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("泛起食指七徽勾一弦", self.opens, context)
        stopped, reason = MODULE.parse_jianzi(
            "泛止名指九徽勾五弦", self.opens, context
        )
        expected, _ = MODULE.position_pitch(5, 9.0, self.opens, "stopped")
        self.assertIsNone(reason)
        self.assertAlmostEqual(stopped[0], expected)
        self.assertFalse(context["harmonic_scope"])
        inherited, reason = MODULE.parse_jianzi("剔五弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertAlmostEqual(inherited[0], expected)

    def test_suffix_fanzhi_keeps_current_note_harmonic_then_ends_region(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("泛起食指七徽勾一弦", self.opens, context)
        harmonic, reason = MODULE.parse_jianzi(
            "名指九徽勾五弦泛止", self.opens, context
        )
        expected, _ = MODULE.position_pitch(5, 9.0, self.opens, "harmonic")
        self.assertIsNone(reason)
        self.assertAlmostEqual(harmonic[0], expected)
        self.assertFalse(context["harmonic_scope"])

    def test_open_midi_does_not_double_apply_tuning_offset(self):
        metadata = {"tuning": {"open_strings": [
            {"string": 1, "pitch": "C", "octave": 3, "semitone_offset": 0},
            {"string": 2, "pitch": "D", "octave": 3, "semitone_offset": 0},
            {"string": 3, "pitch": "F", "octave": 3, "semitone_offset": 0},
            {"string": 4, "pitch": "G", "octave": 3, "semitone_offset": 0},
            # 紧五弦：pitch/octave 已表示实际 B-flat 3；+1 仅保留为调弦溯源。
            {"string": 5, "pitch": "B♭", "octave": 3, "semitone_offset": 1},
            {"string": 6, "pitch": "C", "octave": 4, "semitone_offset": 0},
            {"string": 7, "pitch": "D", "octave": 4, "semitone_offset": 0},
        ]}}
        self.assertEqual(MODULE.parse_open_midi(metadata), [48, 50, 53, 55, 58, 60, 62])

    def test_zhuaqi_releases_previous_thumb_stopped_string_to_open_pitch(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("大指九徽勾5弦", self.opens, context)
        pitches, reason = MODULE.parse_jianzi("爪起", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [57])
        self.assertIsNone(context["active_left_string"])
        self.assertIsNone(context["active_left_hui"])

    def test_zhuaqi_can_sound_with_another_open_string(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("大指九徽勾5弦", self.opens, context)
        pitches, reason = MODULE.parse_jianzi(
            "散挑7弦爪起", self.opens, context
        )
        self.assertIsNone(reason)
        self.assertEqual(pitches, [62, 57])

    def test_zhuaqi_requires_previous_thumb_stopped_note(self):
        context = MODULE.new_context()
        pitches, reason = MODULE.parse_jianzi("爪起", self.opens, context)
        self.assertEqual(pitches, [])
        self.assertEqual(
            reason, "zhuaqi_requires_previous_thumb_stopped_note"
        )

    def test_daiqi_tongsheng_releases_previous_thumb_and_sounds_right_hand_note(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("大指九徽勾5弦", self.opens, context)
        pitches, reason = MODULE.parse_jianzi(
            "散挑7弦带起同声", self.opens, context
        )
        self.assertIsNone(reason)
        self.assertEqual(pitches, [62, 57])
        self.assertIsNone(context["active_left_string"])
        self.assertNotIn("大指", context["left_positions"])

    def test_daiqi_can_release_previous_middle_finger(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("中指九徽勾5弦", self.opens, context)
        pitches, reason = MODULE.parse_jianzi("带起", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [57])
        self.assertNotIn("中指", context["left_positions"])

    def test_mixed_cuo_records_named_stopped_finger_for_daiqi(self):
        for text in ['撮（中指七徽一弦按音＋六弦散音）',
                     '撮（六弦散音＋中指一弦七徽按音）']:
            with self.subTest(text=text):
                context = MODULE.new_context()
                MODULE.parse_jianzi('食指九徽勾七弦', self.opens, context)
                context['left_finger'] = None
                MODULE.parse_jianzi(text, self.opens, context)
                self.assertEqual(context['left_finger'], '中指')
                self.assertEqual(context['left_positions']['中指']['string'], 1)
                self.assertEqual(MODULE.parse_jianzi('带起', self.opens, context), ([48], None))

    def test_other_open_string_does_not_lose_latest_stopped_finger(self):
        context = MODULE.new_context()
        # 名指 existed before 食指 in the cache; updating it must renew recency.
        MODULE.parse_jianzi('名指九徽挑一弦', self.opens, context)
        MODULE.parse_jianzi('食指九徽挑七弦', self.opens, context)
        MODULE.parse_jianzi('名指十徽挑六弦', self.opens, context)
        MODULE.parse_jianzi('散勾三弦', self.opens, context)
        self.assertEqual(context['left_positions']['名指']['string'], 6)
        self.assertEqual(MODULE.parse_jianzi('带起', self.opens, context), ([60], None))

    def test_open_same_string_releases_its_cached_left_position(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi('名指十徽挑六弦', self.opens, context)
        MODULE.parse_jianzi('散勾六弦', self.opens, context)
        self.assertEqual(MODULE.parse_jianzi('带起', self.opens, context),
                         ([], 'daiqi_requires_previous_stopped_note'))

    def test_explicit_daiqi_finger_without_position_does_not_use_other_finger(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi('食指九徽挑七弦', self.opens, context)
        self.assertEqual(MODULE.parse_jianzi('中指带起', self.opens, context),
                         ([], 'daiqi_requires_previous_stopped_note'))

    def test_daiqi_without_stopped_context_is_unresolved(self):
        pitches, reason = MODULE.parse_jianzi("散挑7弦带起同声", self.opens)
        self.assertEqual(pitches, [])
        self.assertEqual(reason, "daiqi_requires_previous_stopped_note")

    def test_li_compact_surface_emits_ordered_open_strings(self):
        for text in ("历四三", "历四、三弦", "厉四三"):
            pitches, reason = MODULE.parse_jianzi(text, self.opens)
            self.assertIsNone(reason)
            self.assertEqual(pitches, [55, 53])

    def test_li_inherits_harmonic_scope_and_hui(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("泛起七徽挑7弦", self.opens, context)
        pitches, reason = MODULE.parse_jianzi("历7六弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [74, 72])
        self.assertEqual(context["right_hand"], "历")

    def test_li_explicit_hui_applies_to_every_swept_string(self):
        pitches, reason = MODULE.parse_jianzi("食指七徽历3二弦", self.opens)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [65, 62])

    def test_marks_harmonic_scope_warning_when_stopped_reading_is_exact(self):
        report = MODULE.audit({
            "metadata": {"tonic": "1=C", "tonic_degree1_midi": 60},
            "open_midi": self.opens,
            "notes": [
                {"index": 1, "jianpu": "2̇", "jianzi": "泛起七徽挑7弦"},
                {"index": 2, "jianpu": "6̣", "jianzi": "中指九徽勾2弦"},
            ],
        }, 50.0)
        detail = report["details"][1]
        self.assertEqual(detail["status"], "mismatched")
        self.assertTrue(detail["possible_harmonic_state_mismatch"])

    def test_li_spans_two_consecutive_score_events(self):
        report = MODULE.audit({
            "metadata": {"tonic": "1=F", "tonic_degree1_midi": 65},
            "open_midi": self.opens,
            "notes": [
                {"index": 38, "jianpu": "2̣", "jianzi": "历四三"},
                {"index": 39, "jianpu": "1̣", "jianzi": ""},
            ],
        }, 50.0)
        self.assertEqual(report["details"][0]["status"], "matched")
        self.assertEqual(report["details"][0]["covered_indices"], [38, 39])

    def test_three_note_li_matches_three_consecutive_attacks(self):
        report = MODULE.audit({
            'metadata': {'tonic': '1=F', 'tonic_degree1_midi': 65},
            'open_midi': self.opens,
            'notes': [
                {'index': 104, 'jianpu': '3̇', 'jianzi': '泛起食指四徽历五四三弦'},
                {'index': 105, 'jianpu': '2̇', 'jianzi': ''},
                {'index': 106, 'jianpu': '1̇', 'jianzi': ''},
            ],
        }, 50.0)
        self.assertEqual(report['details'][0]['status'], 'matched')
        self.assertEqual(report['details'][0]['covered_indices'], [104, 105, 106])

    def test_li_crosses_barline_but_reports_only_wrong_first_pitch(self):
        report = MODULE.audit({
            'metadata': {'tonic': '1=F'},
            'open_midi': self.opens,
            'notes': [
                {'index': 343, 'jianpu': '3', 'jianzi': '大指六徽四分历五四弦'},
                {'index': 344, 'jianpu': '|', 'abc': '|', 'jianzi': ''},
                {'index': 345, 'jianpu': '3', 'jianzi': ''},
            ],
        }, 50.0)
        detail = report['details'][0]
        self.assertEqual(detail['covered_indices'], [343, 345])
        self.assertEqual(detail['status'], 'mismatched')
        self.assertNotEqual(detail.get('reason'), 'li_pitch_count_mismatch')
        self.assertAlmostEqual(detail['pairs'][0]['absolute_cents'], 221.309, places=2)
        self.assertLess(detail['pairs'][1]['absolute_cents'], 50)

    def test_li_extra_note_cannot_cross_extension_and_barline(self):
        report = MODULE.audit({
            'metadata': {'tonic': '1=F', 'tonic_degree1_midi': 65},
            'open_midi': self.opens,
            'notes': [
                {'index': 137, 'jianpu': '3̇', 'jianzi': '泛起大指八徽历五四三弦'},
                {'index': 138, 'jianpu': '3̇', 'jianzi': ''},
                {'index': 139, 'jianpu': '－（延音）', 'jianzi': ''},
                {'index': 140, 'jianpu': '|', 'jianzi': ''},
                {'index': 141, 'jianpu': '5̇', 'jianzi': '挑六弦十徽'},
            ],
        }, 50.0)
        detail = report['details'][0]
        self.assertEqual(detail['status'], 'mismatched')
        self.assertEqual(detail['reason'], 'li_pitch_count_mismatch')
        self.assertEqual(detail['covered_indices'], [137, 138])
        self.assertEqual(len(detail['actual_midis']), 3)
        self.assertEqual(len(detail['expected_midis']), 2)

    def test_prefix_flat_tonic_is_accepted(self):
        self.assertEqual(
            MODULE.parse_tonic_midi({"tonic": "1=bE"}),
            MODULE.parse_tonic_midi({"tonic": "1=Eb"}),
        )

    def test_explicit_b_flat_uses_ordinary_register(self):
        # Only the App's legacy bare ``1=B`` label is pinned to Bb3.  An
        # explicit flat spelling is ordinary Bb4 unless score metadata carries
        # a capture-specific tonic_degree1_midi override.
        self.assertEqual(MODULE.parse_tonic_midi({"tonic": "1=bB"}), 70)
        self.assertEqual(MODULE.parse_tonic_midi({"tonic": "1=Bb"}), 70)

    def test_harmonic_mode_keeps_hui(self):
        context = {
            "harmonic": False, "harmonic_hui": None,
            "stopped_string": None, "stopped_hui": None,
        }
        first, reason = MODULE.parse_jianzi(
            "泛起名指十二徽挑5弦", self.opens, context
        )
        self.assertIsNone(reason)
        self.assertEqual(first, [88])
        second, reason = MODULE.parse_jianzi("勾4弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(second, [86])

    def test_stopped_position_is_inherited_on_same_string(self):
        context = {
            "harmonic": False, "harmonic_hui": None,
            "stopped_string": None, "stopped_hui": None,
        }
        MODULE.parse_jianzi("大指六徽四分挑7弦", self.opens, context)
        pitches, reason = MODULE.parse_jianzi("挑7弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertAlmostEqual(pitches[0], 76.0, delta=0.5)

    def test_stopped_position_is_inherited_on_another_string(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("大指七徽挑4弦", self.opens, context)
        pitches, reason = MODULE.parse_jianzi("勾5弦", self.opens, context)
        expected, _ = MODULE.position_pitch(5, 7, self.opens)
        self.assertIsNone(reason)
        self.assertAlmostEqual(pitches[0], expected, places=4)

    def test_slide_emits_endpoint_pitch_and_updates_context(self):
        context = {
            "harmonic": False, "harmonic_hui": None,
            "stopped_string": None, "stopped_hui": None,
        }
        MODULE.parse_jianzi("大指六徽四分挑7弦", self.opens, context)
        slide, reason = MODULE.parse_jianzi("注下七徽", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(slide, [74])
        pitches, reason = MODULE.parse_jianzi("挑7弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [74])

    def test_plain_movement_updates_position_for_jiu(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("大指九徽勾4弦", self.opens, context)
        slide, reason = MODULE.parse_jianzi("上七徽六分", self.opens, context)
        self.assertIsNone(reason)
        expected_slide, _ = MODULE.position_pitch(4, 7.6, self.opens)
        self.assertAlmostEqual(slide[0], expected_slide)
        pitches, reason = MODULE.parse_jianzi("就挑5弦", self.opens, context)
        self.assertIsNone(reason)
        expected, _ = MODULE.position_pitch(5, 7.6, self.opens)
        self.assertAlmostEqual(pitches[0], expected)

    def test_hu_slide_emits_endpoint_on_inherited_string(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("大指九徽勾4弦", self.opens, context)
        pitches, reason = MODULE.parse_jianzi(
            "浒上七徽九分", self.opens, context
        )
        expected, _ = MODULE.position_pitch(4, 7.9, self.opens)
        self.assertIsNone(reason)
        self.assertAlmostEqual(pitches[0], expected)
        self.assertEqual(context["active_left_string"], 4)
        self.assertEqual(context["active_left_hui"], 7.9)

    def test_other_explicit_slide_types_emit_their_endpoint(self):
        for text, hui in (
            ("绰上五徽六分", 5.6),
            ("注下七徽九分", 7.9),
            ("淌九徽", 9.0),
            ("引上七徽六分", 7.6),
            ("进复七徽", 7.0),
            ("退复九徽", 9.0),
        ):
            with self.subTest(text=text):
                context = MODULE.new_context()
                MODULE.parse_jianzi("大指九徽勾4弦", self.opens, context)
                pitches, reason = MODULE.parse_jianzi(text, self.opens, context)
                expected, _ = MODULE.position_pitch(4, hui, self.opens)
                self.assertIsNone(reason)
                self.assertAlmostEqual(pitches[0], expected)
                self.assertEqual(context["active_left_hui"], hui)

    def test_explicit_pre_attack_chuo_is_pitch_audited(self):
        data = {
            "metadata": {
                "tonic": "1=C",
                "tuning": {"open_strings": [
                    {"string": index, "pitch": pitch, "octave": octave}
                    for index, pitch, octave in (
                        (1, "C", 3), (2, "D", 3), (3, "F", 3), (4, "G", 3),
                        (5, "A", 3), (6, "C", 4), (7, "D", 4),
                    )
                ]},
            },
            "notes": [{"index": 0, "jianpu": "1", "jianzi": "绰大指五徽六分挑6弦"}],
        }
        report = MODULE.audit(data, 50.0)
        self.assertEqual(report["details"][0]["status"], "mismatched")
        self.assertEqual(len(report["details"][0]["actual_midis"]), 1)

    def test_inherited_chuo_and_zhu_have_scalar_pitch_audits(self):
        data = {
            "metadata": {"tonic": "1=C"},
            "open_midi": self.opens,
            "notes": [
                {"index": 0, "jianpu": "1̇", "jianzi": "大指七徽挑六弦"},
                {"index": 1, "jianpu": "7", "jianzi": "注下七徽三分"},
                {"index": 2, "jianpu": "3", "jianzi": "名指九徽勾五弦"},
                {"index": 3, "jianpu": "6", "jianzi": "绰上七徽"},
            ],
        }
        report = MODULE.audit(data, 50.0)
        self.assertEqual(
            [detail["status"] for detail in report["details"]],
            ["matched", "matched", "matched", "matched"],
        )
        self.assertAlmostEqual(
            report["details"][1]["pairs"][0]["jianzi_midi"], 71.0, places=1
        )
        self.assertAlmostEqual(report["details"][3]["pairs"][0]["jianzi_midi"], 69.0)

    def test_jiu_requires_a_current_stopped_position(self):
        pitches, reason = MODULE.parse_jianzi(
            "就挑5弦", self.opens, MODULE.new_context()
        )
        self.assertEqual(pitches, [])
        self.assertEqual(reason, "current_stopped_position_missing")

    def test_open_mode_is_inherited_by_neighbouring_pluck(self):
        context = MODULE.new_context()
        first, reason = MODULE.parse_jianzi("散挑7弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(first, [62])
        second, reason = MODULE.parse_jianzi("勾5弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(second, [57])

    def test_explicit_stopped_note_ends_open_inheritance(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("散挑7弦", self.opens, context)
        stopped, reason = MODULE.parse_jianzi(
            "大指九徽勾5弦", self.opens, context
        )
        self.assertIsNone(reason)
        self.assertNotEqual(stopped, [57])
        inherited, reason = MODULE.parse_jianzi("挑5弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertAlmostEqual(inherited[0], stopped[0])

    def test_fanzhi_ends_harmonic_scope(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("泛起名指十二徽挑5弦", self.opens, context)
        MODULE.parse_jianzi("泛止", self.opens, context)
        pitches, reason = MODULE.parse_jianzi("勾4弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [55])

    def test_rest_with_fanqi_keeps_harmonic_scope(self):
        data = {
            "metadata": {
                "tonic": "1=C",
                "tuning": {"open_strings": [
                    {"string": index, "pitch": pitch, "octave": octave}
                    for index, pitch, octave in (
                        (1, "C", 3), (2, "D", 3), (3, "F", 3),
                        (4, "G", 3), (5, "A", 3), (6, "C", 4), (7, "D", 4),
                    )
                ]},
            },
            "notes": [
                {"index": 0, "jianpu": "0（休止）", "jianzi": "泛起名指十二徽挑5弦"},
                {"index": 1, "jianpu": "1", "jianzi": "勾4弦"},
            ],
        }
        row = MODULE.audit(data, 50)["details"][1]
        self.assertNotEqual(row.get("reason"), "harmonic_hui_missing")

    def test_bare_rest_ends_prior_harmonic_scope(self):
        report = MODULE.audit({
            "metadata": {"tonic": "1=C", "tonic_degree1_midi": 60},
            "open_midi": self.opens,
            "notes": [
                {"index": 0, "jianpu": "5", "jianzi": "泛起名指九徽挑5弦"},
                {"index": 1, "jianpu": "0（休止）", "jianzi": ""},
                {"index": 2, "jianpu": "3", "jianzi": "名指九徽勾5弦"},
            ],
        }, 50.0)
        self.assertEqual(report["details"][2]["status"], "matched")



    def test_single_symbol_cuo_passes_on_main_note(self):
        # 同音加厚：谱字 1 (=60)，撮出 四弦九徽(≈60)＋六弦散音(60) —— 主音匹配即过。
        report = MODULE.audit({
            "metadata": {"tonic": "1=C", "tonic_degree1_midi": 60},
            "open_midi": self.opens,
            "notes": [
                {"index": 1, "jianpu": "1",
                 "jianzi": "撮（大指九徽四弦按音＋六弦散音）"},
            ],
        }, 50.0)
        detail = report["details"][0]
        self.assertEqual(detail["status"], "matched")
        self.assertEqual(detail.get("reason"), "single_symbol_multivoice_main_note")

    def test_single_symbol_octave_cuo_passes_on_main_note(self):
        # 八度省略记谱：谱字 1 (=60)，撮出 六弦散音(60)＋一弦散音(48)。
        report = MODULE.audit({
            "metadata": {"tonic": "1=C", "tonic_degree1_midi": 60},
            "open_midi": self.opens,
            "notes": [
                {"index": 1, "jianpu": "1",
                 "jianzi": "撮（六弦散音＋一弦散音）"},
            ],
        }, 50.0)
        detail = report["details"][0]
        self.assertEqual(detail["status"], "matched")
        self.assertEqual(detail.get("reason"), "single_symbol_multivoice_main_note")

    def test_yinghe_with_two_written_pitches_passes_if_one_pitch_matches(self):
        report = MODULE.audit({
            "metadata": {"tonic": "1=G"},
            "open_midi": self.opens,
            "notes": [{
                "index": 323,
                "jianpu": "1̣",
                "jianpu_alt": "1",
                "jianzi": "大指七徽六分勾五弦应合",
            }],
        }, 50.0)
        detail = report["details"][0]
        self.assertEqual(detail["status"], "matched")
        self.assertEqual(detail["reason"], "yinghe_single_pitch_match")
        self.assertEqual(detail["expected_pitch_count"], 2)
        self.assertEqual(detail["actual_pitch_count"], 1)
        self.assertLessEqual(detail["pairs"][0]["absolute_cents"], 50.0)

    def test_yinghe_still_warns_when_neither_written_pitch_matches(self):
        report = MODULE.audit({
            "metadata": {"tonic": "1=G"},
            "open_midi": self.opens,
            "notes": [{
                "index": 323,
                "jianpu": "2̣",
                "jianpu_alt": "3",
                "jianzi": "大指七徽六分勾五弦应合",
            }],
        }, 50.0)
        detail = report["details"][0]
        self.assertEqual(detail["status"], "mismatched")
        self.assertEqual(detail["reason"], "yinghe_no_pitch_match")

    def test_fanghe_accepts_one_match_and_ruyi_inherits_first_string_position(self):
        report = MODULE.audit({
            "metadata": {"tonic": "1=C"},
            "open_midi": self.opens,
            "notes": [
                {"index": 58, "jianpu": "5̣", "jianpu_alt": "5",
                 "jianzi": "名指七徽六分勾五弦放合"},
                {"index": 59, "jianpu": "5̣", "jianpu_alt": "5",
                 "jianzi": "剔五弦散如一四弦"},
            ],
        }, 50.0)
        self.assertEqual(
            [row["status"] for row in report["details"]],
            ["matched", "matched"],
        )
        self.assertEqual(report["details"][0]["reason"], "fanghe_single_pitch_match")
        self.assertIsNone(report["details"][1].get("reason"))
        self.assertEqual(
            [pair["absolute_cents"] for pair in report["details"][1]["pairs"]],
            [0.0, 3.802],
        )

    def test_fanghe_still_warns_when_neither_pitch_matches(self):
        report = MODULE.audit({
            "metadata": {"tonic": "1=C"},
            "open_midi": self.opens,
            "notes": [{
                "index": 1, "jianpu": "2̣", "jianpu_alt": "3",
                "jianzi": "名指七徽六分勾五弦放合",
            }],
        }, 50.0)
        detail = report["details"][0]
        self.assertEqual(detail["status"], "mismatched")
        self.assertEqual(detail["reason"], "fanghe_no_pitch_match")

    def test_ruyi_requires_the_inherited_and_open_pitches_to_match(self):
        report = MODULE.audit({
            "metadata": {"tonic": "1=C"},
            "open_midi": self.opens,
            "notes": [
                {"index": 1, "jianpu": "5",
                 "jianzi": "名指七徽六分勾五弦"},
                {"index": 2, "jianpu": "2̣", "jianpu_alt": "3",
                 "jianzi": "剔五弦散如一四弦"},
            ],
        }, 50.0)
        detail = report["details"][1]
        self.assertEqual(detail["status"], "mismatched")
        self.assertTrue(any(pair["absolute_cents"] > 50 for pair in detail["pairs"]))

    def test_single_symbol_cuo_with_no_matching_member_still_fails(self):
        # 两个成员都不匹配谱字（5=67 vs 60/48）：仍按数量不匹配报错。
        report = MODULE.audit({
            "metadata": {"tonic": "1=C", "tonic_degree1_midi": 60},
            "open_midi": self.opens,
            "notes": [
                {"index": 1, "jianpu": "5",
                 "jianzi": "撮（大指九徽四弦按音＋一弦散音）"},
            ],
        }, 50.0)
        detail = report["details"][0]
        self.assertEqual(detail["status"], "mismatched")
        self.assertEqual(detail.get("reason"), "pitch_count_mismatch")

    def test_dual_symbol_cuo_keeps_pairwise_matching(self):
        # 双谱字仍维持 2v2 最优配对，不因新规则放宽。
        report = MODULE.audit({
            "metadata": {"tonic": "1=C", "tonic_degree1_midi": 60},
            "open_midi": self.opens,
            "notes": [
                {"index": 1, "jianpu": "4̣", "jianpu_alt": "6",
                 "jianzi": "撮（大指七徽九分六弦按音＋三弦散音）"},
            ],
        }, 50.0)
        detail = report["details"][0]
        self.assertEqual(detail["status"], "matched")
        self.assertIsNone(detail.get("reason"))


if __name__ == "__main__":
    unittest.main()
