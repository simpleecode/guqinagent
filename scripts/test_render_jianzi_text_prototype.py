"""Local renderer regressions; does not assert App-identical geometry."""
import unittest
from pathlib import Path
from PIL import ImageChops, Image
from render_jianzi_text_prototype import encode, render, position, COMPOUNDS

FONT=Path(__file__).resolve().parents[1]/'work/jianzi_renderer_probe/fonts/font_1.ttf'


class EncodingTests(unittest.TestCase):
    def test_basic_roundtrip(self):
        std,p=encode('大指七徽勾四弦')
        self.assertEqual(std,{'tp':'zhyx','a':'d','b':'7','d':'g','e':'4'})
        self.assertEqual(p['decoded_cn'],'大指七徽勾4弦')

    def test_positions(self):
        for text,compact in [('七徽六分','76'),('十徽二分','X02'),
                             ('十一徽','X1'),('十二徽二分','X22'),('十三徽','X3'),('徽外','Y')]:
            with self.subTest(text=text):self.assertEqual(position(text)['compact'],compact)

    def test_two_independent_hui(self):
        std,p=encode('撮（大指七徽六弦按音＋食指九徽一弦按音）')
        self.assertEqual(std['raw_fields']['off_20'],'9')
        self.assertEqual(p['decoded_cn'],'撮（大指七徽6弦按音＋食指九徽1弦按音）')

    def test_open_first(self):
        _,p=encode('撮（一弦散音＋名指十二徽二分六弦按音）')
        self.assertEqual(p['decoded_cn'],'撮（1弦散音＋名指十二徽二分6弦按音）')

    def test_open_open_and_families(self):
        for tech in COMPOUNDS:
            with self.subTest(tech=tech):
                std,p=encode(f'{tech}（一弦散音＋七弦散音）')
                self.assertEqual(std['a'],COMPOUNDS[tech])
                self.assertIn('1弦散音',p['decoded_cn'])
                self.assertIn('7弦散音',p['decoded_cn'])

    def test_harmonic_scope_is_explicit(self):
        _,p=encode('泛音撮（大指七徽六弦泛音＋名指九徽一弦泛音）')
        self.assertFalse(p['app_encoding_complete'])
        self.assertTrue(p['harmonic'])
        self.assertEqual([n['tone'] for n in p['branches']],['harmonic','harmonic'])

    def test_omission_not_invented(self):
        _,p=encode('挑五弦')
        self.assertIsNone(p['finger']);self.assertIsNone(p['position'])

    def test_slide(self):
        _,p=encode('注下七徽六分')
        self.assertEqual(p['decoded_cn'],'注下七徽六分')

    def test_reject_invalid(self):
        for text in ['大指十四徽勾四弦','大指七徽勾八弦','大指勾四弦',
                     '散大指七徽勾四弦','撮（一弦散音＋一弦散音）','未知指法',
                     '撮（一弦散音＋六弦散音＋七弦散音）']:
            with self.subTest(text=text),self.assertRaises(ValueError):encode(text)

    @unittest.skipUnless(FONT.exists(),'Recovered local font not present')
    def test_images_not_clipped(self):
        examples=['大指七徽勾四弦','名指九徽挑五弦','中指五徽剔七弦',
                  '大指十二徽二分挑七弦','泛音名指七徽勾四弦','绰上七徽六分',
                  '撮（大指七徽六弦按音＋食指九徽一弦按音）']
        for text in examples:
            with self.subTest(text=text):
                std,p=encode(text);image=render(std,p,FONT)
                bbox=ImageChops.difference(image,Image.new('RGB',image.size,'white')).getbbox()
                self.assertIsNotNone(bbox)
                self.assertGreater(bbox[0],0);self.assertGreater(bbox[1],0)
                self.assertLess(bbox[2],320);self.assertLess(bbox[3],320)


if __name__=='__main__':unittest.main()
