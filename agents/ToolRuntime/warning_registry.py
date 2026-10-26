"""Shared warning registry."""

INHERITED_GESTURE_WARNING_CODES = {
    'tiao_on_first_string',
    'slide_attack_missing_left_finger',
    'same_position_finger_change',
    'jinfu_without_downward_walk',
    'open_attack_ornament',
    'li_nonadjacent_strings',
    'thumb_lower_left_position',
    'open_string_unplayable_gesture',
    'kneeling_taoqi_below_seventh_hui',
    'taoqi_without_thumb_origin',
    'repeated_fu_without_walk',
    'fu_without_walk_origin',
    'harmonic_ornament_without_stopped',
}

NUMBERED_REPEAT_PATTERN = r'从\s*(?:音序\s*)?(\d+)\s*(?:[（(]音序[）)]\s*)?再[作做]'

MISSING_REPEAT_ANCHOR_TEXT = '缺少再作标记（“从ㄱ再作”前需先有“[再作标记ㄱ]”；也可改用“从n再作”，n为匹配的历史音序）'

COMPOUND_REPEAT_TEXT = ('掐撮三声不应与再作指令合写（请删去本行再作；'
                        '由掐撮三声覆盖的后续音直接置空，不用再作省略标记）')

MISSING_ATTACK_TEXT = "独立发音缺少减字（前序为普通单次起音，不能覆盖此音；请填写起音指法）"
ANCHOR_BLANK_TEXT = ('再作标记后不能直接连续置空（“再作标记ㄱ”只标记重复起点；'
                     '在实际重复处用“从ㄱ再作”，否则请补全后续减字）')

# Public score advisories; protocol errors and private reference checks are excluded.
PUBLIC_WARNING_CODES = INHERITED_GESTURE_WARNING_CODES | {
    'repeat_anchor_followed_by_blanks',
    'repeat_anchor_pitch_mismatch',
    'compound_repeat_instruction_overlap',
    'consecutive_walks_without_attack',
    'fractional_hui_in_harmonic',
    'harmonic_to_same_string_attack',
    'illegal_blank_after_walk',
    'invalid_string_number',
    'jianzi_pitch_mismatch',
    'low_hui_stopped_too_cramped',
    'missing_independent_attack',
    'missing_repeat_anchor',
    'missing_repeat_instruction',
    'nonadjacent_pluck_pair',
    'open_to_stopped_same_string',
    'ordinary_open_same_string_reattack',
    'repeat_anchor_instruction_overlap',
    'repeat_from_head_pitch_mismatch',
    'repeat_plain_pitch_mismatch',
    'zhi_missing_continuous_gesture',
    'zhi_sweep_pitch_mismatch',
    'finite_gesture_uncovered_note',
    'repeat_numbered_pitch_mismatch',
    'repeat_numbered_start_missing',
    'repeat_omission_marker_missing',
    'same_finger_reach_too_large',
    'same_string_cuo_pair',
    'walk_direction_reversed',
    'walk_endpoint_same_as_start',
    'walk_inside_harmonic_region',
    'walk_span_too_large',
}
