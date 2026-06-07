#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2026 modelbest
#
# @date: 2026
#

caption_en = [
    'Describe the image concisely',
    'Provide a brief description of the given image',
    'Offer a succinct explanation of the picture presented',
    'Summarize the visual content of the image',
    'Share a conciseinter pretation of the image provided',
    'Present a compact description of the photo’s key features',
    'Relay a brief and clear account of the picture shown',
    'Render a clear and concise summary of the photo',
    'Write a terse but informative summary of the picture',
    'Create a compact narrative representing the image presented',
]

caption_zh = [
    '简明扼要地描述图像',
    '提供给定图像的简短描述',
    '对所示的图片进行简要的解释',
    '总结图像的视觉内容',
    '对所提供的图像进行简要的解释',
    '简明扼要并清楚地说明所示图片',
    '对这张照片作一个简明扼要的总结',
    '写一篇简洁但内容丰富的图片摘要',
    '创造一个紧凑的叙事来代表所呈现的图像',
]


PROMPT_FOR_PPT = (
    "Please analyze the slide screenshot based on the following steps.\n"
    "1. Find the title, slide header, slide footer and page number of the slide.\n"
    "2. Give a concise overview of slide screenshot.\n"
    "3. Besides the title and footer, analyze the number of charts included in the screenshot. For each section, you also need to output its type, position and introduction.\n"
    "4. For each section, analyze the detail and give a comprehensive description of all you can see."
)


detailed_instructions = [
    "Identify and describe each object in the image in detail.",
    "Describe the key features of the image in great detail.",
    "What are the main elements in this image? Describe them thoroughly.",
    "Explain what's happening in the image with as much detail as possible.",
    "Detail the image's components with particular focus on each entity.",
    "Provide an intricate description of every entity in the image.",
    "What are the main objects or subjects in the image? Please describe them in detail.",
    "What is the setting or environment in which the image takes place?",
    "How do the elements in the image relate to each other in terms of positioning or composition?",
    "Explain the elements of the image with thorough attention to detail.",
    "Explain the image's various components in depth.",
    "What are the key features you observe in the image?",
    "Can you point out the details that make this image unique?",
    "Itemize the elements you identify in the image and describe them thoroughly.",
    "Convey the specifics of the image with meticulous attention to detail.",
    "Tell me what catches your eye in the image, and describe those elements in depth.",
]


detailed_instructions_zh = [
    "请详细识别并描述图像中的每个物体。",
    "请详细描述图像的关键特征。",
    "图像中的主要元素是什么？请详细描述它们。",
    "请尽可能详细地解释图像中的元素。",
    "详细描述图像的组成部分，关注图像中的每个实体。",
    "提供图像中每个实体的详细描述。",
    "图像中的主要物体或主体是什么？请详细描述它们。",
    "详细介绍图像中的所有元素。",
    "请详细解释图像的各个元素。",
    "深入解释图像的各种组成部分。",
    "你在图像中观察到哪些关键特征？",
    "你能指出使这张图像独特的细节吗？",
    "列出你在图像中识别的元素并详细描述它们。",
    "详细描述图像的具体细节。",
    "请对图像中吸引你注意的地方进行详细描述。",
]