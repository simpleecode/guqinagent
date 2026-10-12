#!/usr/bin/env python3
"""Natural Jianzi -> App fields -> bounds-aware local glyph composition.

Basic enum is verified; compound/slide glyph identification and all geometry
remain local prototypes, not pixel-identical App rendering. See encoding.json.
"""
import argparse
import json
import re
import math
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from extract_jianpu_jianzi import LEFT_FINGER, RIGHT_TECH, decode_jianzi_glyph

FINGERS={value:key for key,value in LEFT_FINGER.items()}
# Basic right-hand enum ordering from libapp pp+0x13320, checked with the font atlas.
RIGHT_CODES={'抹':'m','挑':'t','勾':'g','剔':'q','打':'d','摘':'z','擘':'b','托':'o'}
FINGER_GLYPH={'d':0x8301,'s':0x8302,'z':0x8303,'m':0x8304,'g':0x8305}
RIGHT_GLYPH={code:0x8201+i for i,code in enumerate(['m','t','g','q','d','z','b','o'])}
DIGITS={c:i for i,c in enumerate('〇一二三四五六七八九')}
COMPOUNDS={'撮':'pc:','泼':'pp:','剌':'pl:','泼剌':'ppl:','反撮':'pfc:',
           '弹':'pt:','双弹':'pet:','三弹':'pst:'}
# Tentative identifications from the atlas: not yet phone-side verified.
COMPOUND_GLYPH={'pc:':0x8701,'pfc:':0x8702,'pp:':0x8603,'pl:':0x8602,
                'ppl:':0x8604,'pt:':0x8706,'pet:':0x8707,'pst:':0x8708}
SLIDES={'绰上':'u:','注下':'d:','进':'j:','退':'t:','淌':'td:',
        '浒上':'hu:','进复':'eu:','退复':'ed:'}
SLIDE_GLYPH={'u:':0x8609,'d:':0x860A,'j:':0x851A,'t:':0x8519,
             'td:':0x8518,'hu:':0x8618,'eu:':0x85AA,'ed:':0x85AB}
NUM=r'[一二三四五六七八九十\d]+'
FINGER=r'(?:大指|食指|中指|名指|跪指)'
HUI=rf'(?:{NUM}徽(?:[一二三四五六七八九\d]分)?|徽外)'


def number(text):
    if text.isdigit():return int(text)
    if text in DIGITS:return DIGITS[text]
    if text=='十':return 10
    if text.startswith('十') and text[1:] in DIGITS:return 10+DIGITS[text[1:]]
    raise ValueError('Unsupported number: '+text)


def position(text):
    if text=='徽外':return {'hui':'outside','fen':None,'compact':'Y'}
    m=re.fullmatch(rf'({NUM})徽(?:([一二三四五六七八九\d])分)?',text)
    if not m:raise ValueError('Invalid hui: '+text)
    hui=number(m[1]);fen=number(m[2]) if m[2] else None
    if not 1<=hui<=13 or (fen is not None and not 1<=fen<=9):raise ValueError('Out-of-range hui/fen')
    compact=('X'+(str(hui-10)+str(fen) if fen else (str(hui-10) if hui>10 else ''))
             if hui>=10 else str(hui)+(str(fen) if fen else ''))
    return {'hui':hui,'fen':fen,'compact':compact}


def string_number(text):
    value=number(text)
    if not 1<=value<=7:raise ValueError('Out-of-range string')
    return value


def branch(text):
    m=re.fullmatch(rf'(?:(泛音)?({FINGER})({HUI}))?([一二三四五六七1-7])弦(按音|泛音|散音)?',text)
    if not m:raise ValueError('Unsupported compound branch: '+text)
    harmonic,finger,hui,string,tone=m.groups()
    if (finger and tone=='散音') or (not finger and tone in ('按音','泛音')):
        raise ValueError('Inconsistent compound tone/finger/hui: '+text)
    return {'finger':FINGERS.get(finger),'position':position(hui) if hui else None,
            'string':string_number(string),'tone':'harmonic' if harmonic or tone=='泛音' else ('stopped' if finger else 'open')}


def encode(text):
    text=re.sub(r'\s+','',text).replace('拨剌','泼剌').replace('拨（','泼（')
    m=re.fullmatch(r'(泛音)?(反撮|泼剌|双弹|三弹|撮|泼|剌|弹)[（\[](.*)[）\]]',text)
    if m:
        harmonic,tech,contents=m.groups();parts=re.split(r'[＋+]',contents)
        if len(parts)!=2:raise ValueError('Compound needs two explicit branches')
        notes=[branch(part) for part in parts]
        if notes[0]['string']==notes[1]['string']:raise ValueError('Compound strings must differ')
        raw={'off_8':COMPOUNDS[tech]}
        for note,f,h,s in zip(notes,['off_c','off_1c'],['off_10','off_20'],['off_14','off_24']):
            raw[f]=note['finger'] or '';raw[h]=note['position']['compact'] if note['position'] else '';raw[s]=str(note['string'])
        std={'tp':None,'a':COMPOUNDS[tech],'b':raw['off_c'],'c':raw['off_10'],
             'd':raw['off_14'],'f':raw['off_1c'],'g':raw['off_24'],'raw_fields':raw}
        if not notes[0]['finger'] and notes[1]['finger']:
            # Open-first/stopped-second uses the modern layout understood by
            # the production decoder. Empty off_c does not select its raw layout.
            std={'tp':None,'a':COMPOUNDS[tech],'d':str(notes[0]['string']),
                 'e':notes[1]['finger'],'f':notes[1]['position']['compact'],
                 'g':str(notes[1]['string'])}
        parsed={'kind':'compound','tech':tech,'branches':notes,'harmonic':bool(harmonic),
                'glyph_mapping_status':'tentative_atlas_identification',
                'encoding_scope':'std stores fingers/hui/strings; harmonic state stays explicit in layout AST'}
    else:
        m=re.fullmatch(rf'(绰上|注下|进复|退复|进|退|淌|浒上)({HUI})',text)
        if m:
            tech,hui=m.groups();pos=position(hui)
            std={'tp':'th','t':SLIDES[tech],'h':pos['compact']}
            parsed={'kind':'slide','position':pos,'tech':tech,'glyph_mapping_status':'tentative_atlas_identification'}
        else:
            m=re.fullmatch(rf'(泛音|散音|散|绰)?({FINGER})?({HUI})?(抹|挑|勾|剔|打|摘|擘|托)([一二三四五六七1-7])弦',text)
            if not m:raise ValueError('Unsupported Jianzi (no text fallback): '+text)
            modifier,finger,hui,right,string=m.groups();pos=position(hui) if hui else None;xian=string_number(string)
            if bool(finger)!=bool(hui):raise ValueError('Specify both finger/hui or omit both')
            if modifier in ('散','散音') and finger:raise ValueError('Open note must not have finger/hui')
            if modifier in ('泛音','绰') and not finger:raise ValueError('Explicit 泛音/绰 requires finger/hui')
            if modifier in ('泛音','绰'):
                std={'tp':'zhtyx','z':FINGERS[finger],'h':pos['compact'],'t':'f:' if modifier=='泛音' else 'c:',
                     'y':RIGHT_CODES[right],'x':str(xian)}
            elif finger:std={'tp':'zhyx','a':FINGERS[finger],'b':pos['compact'],'d':RIGHT_CODES[right],'e':str(xian)}
            else:
                std={'tp':'tyx' if modifier else 'yx','d':RIGHT_CODES[right],'e':str(xian)}
                if modifier:std['c']='s:'
            parsed={'kind':'single','finger':FINGERS.get(finger),'position':pos,'right':RIGHT_CODES[right],
                    'string':xian,'modifier':modifier,'glyph_mapping_status':'basic_enum_verified; modifiers_atlas_identified'}
    parsed['decoded_cn']=decode_jianzi_glyph({'std':std})['cn']
    parsed['app_encoding_complete']=not (parsed['kind']=='compound' and
        (parsed['harmonic'] or any(note['tone']=='harmonic' for note in parsed['branches'])))
    return std,parsed


@lru_cache(maxsize=256)
def glyph_mask(font_path,cp):
    font=ImageFont.truetype(font_path,512);ch=chr(cp);left,top,right,bottom=font.getbbox(ch)
    mask=Image.new('L',(max(1,right-left),max(1,bottom-top)),0)
    ImageDraw.Draw(mask).text((-left,-top),ch,font=font,fill=255)
    bbox=mask.getbbox()
    if not bbox:raise ValueError(f'Empty glyph U+{cp:04X}')
    return mask.crop(bbox)


def put(image,font_path,cp,box):
    mask=glyph_mask(str(font_path),cp);x,y,w,h=box;scale=min(w/mask.width,h/mask.height)
    mask=mask.resize((max(1,round(mask.width*scale)),max(1,round(mask.height*scale))),Image.Resampling.LANCZOS)
    image.paste((0,0,0),(round(x+(w-mask.width)/2),round(y+(h-mask.height)/2)),mask)


def hui_draw(image,pos,font,box):
    x,y,w,h=box;hui=pos['hui'];fen=pos['fen']
    digits=[0x840B] if hui=='outside' else ([0x840A,0x8400+hui-10] if hui>10 else [0x8400+hui])
    if len(digits)==1 and not fen:put(image,font,digits[0],box)
    else:
        for i,cp in enumerate(digits):put(image,font,cp,(x,y+i*h*.42,w*.68,h*.42))
        if fen:put(image,font,0x8400+fen,(x+w*.55,y+h*.57,w*.45,h*.40))


def header(image,finger,pos,font,box):
    x,y,w,h=box
    if finger:
        put(image,font,FINGER_GLYPH[finger],(x,y,w*.45,h))
        hui_draw(image,pos,font,(x+w*.52,y,w*.48,h))


def render(std,parsed,font_path):
    image=Image.new('RGB',(320,320),'white')
    if parsed['kind']=='single':
        modifier=parsed['modifier'];x,w=(58,238) if modifier in ('泛音','绰') else (34,252)
        if modifier in ('泛音','绰'):put(image,font_path,0x8501 if modifier=='泛音' else 0x8505,(12,42,36,67))
        if parsed['finger']:header(image,parsed['finger'],parsed['position'],font_path,(x+12,22,w-24,82))
        y=119 if parsed['finger'] else 72;right=parsed['right']
        put(image,font_path,RIGHT_GLYPH[right],(x,y,w,174))
        # Distinct ink cavities for 勾/挑/抹 etc.; not a shared font anchor.
        cavities={'g':(.30,.35,.50,.31),'q':(.30,.35,.50,.31),
                  't':(.28,.14,.58,.35),'o':(.30,.15,.55,.32),
                  'm':(.30,.48,.43,.30),'b':(.32,.35,.44,.30),
                  'd':(.25,.35,.44,.30),'z':(.30,.35,.45,.30)}
        a,b,c,d=cavities[right]
        put(image,font_path,0x8100+parsed['string'],(x+w*a,y+174*b,w*c,174*d))
    elif parsed['kind']=='slide':
        put(image,font_path,SLIDE_GLYPH[std['t']],(34,38,252,95))
        hui_draw(image,parsed['position'],font_path,(88,161,144,125))
    else:
        for i,note in enumerate(parsed['branches']):
            x=28+i*140
            if note['finger']:header(image,note['finger'],note['position'],font_path,(x,22,124,83))
            if note['tone']=='harmonic' or parsed['harmonic']:put(image,font_path,0x8501,(x,114,32,35))
            put(image,font_path,0x8100+note['string'],(x+27,154,88,51))
        put(image,font_path,COMPOUND_GLYPH[std['a']],(78,218,164,78))
    return image


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--text',action='append',required=True)
    p.add_argument('--font',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--columns',type=int,default=3)
    args=p.parse_args()
    if args.columns<1:p.error('--columns must be positive')
    cmap=TTFont(args.font).getBestCmap()
    assert 0x8203 in cmap and 0x8301 in cmap and 0x8407 in cmap
    inputs=[(text,*encode(text)) for text in args.text]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cols=min(args.columns,len(inputs));preview=Image.new('RGB',(360*cols,400*math.ceil(len(inputs)/cols)),'white')
    records=[];label=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Unicode.ttf',16)
    for i,(text,std,parsed) in enumerate(inputs,1):
        image=render(std,parsed,args.font);image.save(args.output_dir/f'jianzi_{i}.png')
        card=Image.new('RGB',(360,400),'white');card.paste(image,(20,10));draw=ImageDraw.Draw(card)
        for j,start in enumerate(range(0,len(text),20)):draw.text((12,342+j*22),text[start:start+20],font=label,fill='black')
        if parsed['kind']!='single':draw.text((12,382),'部件映射／布局待 App 核验',font=label,fill='#777777')
        preview.paste(card,(((i-1)%cols)*360,((i-1)//cols)*400))
        records.append({'text':text,'app_std':std,**parsed,'image':f'jianzi_{i}.png','layout_status':'bounds_aware_local_layout_not_app_verified'})
    preview.save(args.output_dir/'preview.png')
    (args.output_dir/'encoding.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(f'Rendered {len(records)} Jianzi: {args.output_dir / "preview.png"}')


if __name__=='__main__':main()
