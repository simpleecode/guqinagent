#!/usr/bin/env python3
"""Inspect recovered Jianzi glyph components; this is not a notation renderer."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont


def main():
    p=argparse.ArgumentParser();p.add_argument('--font',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();codes=[c for c in TTFont(args.font).getBestCmap() if c>32]
    columns=10;cw,ch=150,145;image=Image.new('RGB',(columns*cw,((len(codes)+columns-1)//columns)*ch),'white')
    draw=ImageDraw.Draw(image);font=ImageFont.truetype(str(args.font),85);label=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',15)
    for i,code in enumerate(codes):
        x=(i%columns)*cw;y=(i//columns)*ch
        draw.rectangle((x,y,x+cw-1,y+ch-1),outline='#ddd')
        draw.text((x+cw/2,y+15),chr(code),font=font,fill='black',anchor='mt')
        draw.text((x+10,y+ch-24),f'U+{code:04X}',font=label,fill='#444')
    args.output.parent.mkdir(parents=True,exist_ok=True);image.save(args.output)
    print(args.output.resolve())


if __name__=='__main__':main()
