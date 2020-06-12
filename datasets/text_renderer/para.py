# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:44:59 2019

@author: hu
"""
import numpy as np
import random

from datasets.text_renderer.libs.config import load_config
import datasets.text_renderer.libs.utils as utils
import datasets.text_renderer.libs.font_utils as font_utils
from datasets.text_renderer.textrenderer.corpus.corpus_utils import corpus_factory
from datasets.text_renderer.textrenderer.renderer import Renderer


class Para:        
    def __init__(self):
        self.length = 10 # char num per img
        self.clip_max_chars = False
        self.img_height = 32
        self.img_width = 256
        self.chars_file = './datasets/text_renderer/data/chars/chars.txt'#tongyong.txt'
        self.config_file = './datasets/text_renderer/configs/regular.yaml'
        self.fonts_list = './datasets/text_renderer/data/fonts/chn/'
        self.bg_dir = './datasets/text_renderer/data/bg'
        self.corpus_dir = "./datasets/corpus/"#"./data/corpus/txt_gen"  # txt file
        self.corpus_mode = 'chn'  # choices=['random', 'chn', 'eng', 'list']
        self.output_dir = './output'
        self.tag = 'default'
        self.debug = False
        self.viz=False
        self.strict = True
        self.gpu = False
        self.num_processes = None

    def random_set_para(self):
        min_char = 10#2
        max_char = 10#13
        self.length = random.randint(min_char, max_char)
        
        min_hw_ratio = 0.8
        max_hw_ratio = 1.0
        ratio = random.uniform(min_hw_ratio, max_hw_ratio)
        
        self.img_width = int(ratio * self.length * self.img_height)
        
        
class GenPara:
    
    flags = Para()   
    cfg = load_config(flags.config_file)    
    fonts = font_utils.get_font_path_from_fileway(flags.fonts_list)    
    bgs = utils.load_bgs(flags.bg_dir)    
    corpus = corpus_factory(flags.corpus_mode, flags.chars_file, flags.corpus_dir, flags.length)
    font_unsupport_chars = font_utils.get_unsupported_chars(fonts, corpus.chars_file)
    renderer = Renderer(corpus, fonts, bgs, cfg,
                        height=flags.img_height,
                        width=flags.img_width,
                        clip_max_chars=flags.clip_max_chars,
                        debug=flags.debug,
                        gpu=flags.gpu,
                        strict=flags.strict, 
                        font_unsupport_chars=font_unsupport_chars)
                        
    def initial_genpara(self):
        self.flags.random_set_para()
        self.corpus.length = self.flags.length
        self.corpus.img_width = self.flags.img_width
        # self.renderer = self.renderer
        self.renderer = Renderer(self.corpus, self.fonts, self.bgs, self.cfg,
                                height=self.flags.img_height,
                                width=self.flags.img_width,
                                clip_max_chars=self.flags.clip_max_chars,
                                debug=self.flags.debug,
                                gpu=self.flags.gpu,
                                strict=self.flags.strict,
                                font_unsupport_chars=self.font_unsupport_chars)        
        
