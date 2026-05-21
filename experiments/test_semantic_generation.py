"""
TESM 语义生成能力测试脚本

使用50个中文句子进行过拟合测试，验证模型的语义理解和生成能力。
分词器基于句子中出现的所有字符构建词表。

用法:
    python test_semantic_generation.py --backend torch
    python test_semantic_generation.py --backend cuda
    python test_semantic_generation.py --backend triton
    python test_semantic_generation.py --backend auto
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tesm_ssm import TESMConfig, TESMLMHeadModel


# ==================== 50个中文测试句子 ====================
# 涵盖多种语义场景：问答、对话、描述、推理等
# 每个句子扩展为详细的长文本，总计约数千字
TEST_SENTENCES = [
    # 字符覆盖类 (5句) - 确保词表覆盖所有数字和字母
    "数字学习：0123456789是十个阿拉伯数字。0表示没有，1表示单个，2表示一对，3表示三个，4表示四个，5表示五个，6表示六个，7表示七个，8表示八个，9表示九个。这些数字可以组合成任意数值，如10、20、30、40、50、60、70、80、90、100。数字是数学计算的基础，如1+1=2，2+2=4，3+3=6，4+4=8，5+5=10。数字也用于计数、排序、编码等各种场景。理解数字的含义和用法是学习数学和编程的基础。",
    
    "小写字母学习：abcdefghijklmnopqrstuvwxyz是26个英文字母的小写形式。a是第一个字母，b是第二个，c是第三个，d是第四个，e是第五个，f是第六个，g是第七个，h是第八个，i是第九个，j是第十个，k是第十一个，l是第十二个，m是第十三个，n是第十四个，o是第十五个，p是第十六个，q是第十七个，r是第十八个，s是第十九个，t是第二十个，u是第二十一个，v是第二十二个，w是第二十三个，x是第二十四个，y是第二十五个，z是第二十六个。",
    
    "大写字母学习：ABCDEFGHIJKLMNOPQRSTUVWXYZ是26个英文字母的大写形式。A是第一个字母，B是第二个，C是第三个，D是第四个，E是第五个，F是第六个，G是第七个，H是第八个，I是第九个，J是第十个，K是第十一个，L是第十二个，M是第十三个，N是第十四个，O是第十五个，P是第十六个，Q是第十七个，R是第十八个，S是第十九个，T是第二十个，U是第二十一个，V是第二十二个，W是第二十三个，X是第二十四个，Y是第二十五个，Z是第二十六个。",
    
    "数学符号学习：加号+表示加法运算，减号-表示减法运算，乘号*表示乘法运算，除号/表示除法运算，等号=表示相等关系。例如：1+2=3，10-3=7，4*5=20，20/4=5。还有小于号<和大于号>用于比较大小，如5<10表示5小于10，10>5表示10大于5。括号()用于改变运算顺序，如(1+2)*3=9。百分号%表示百分比，如50%等于0.5。这些符号是数学表达的基本元素。",
    
    "编程符号学习：编程中常用的符号包括等号=用于赋值，双等号==用于判断相等，感叹号!表示非，不等于!=。大括号{}用于定义代码块，中括号[]用于数组索引，小括号()用于函数调用。冒号:用于切片和字典，分号;用于语句分隔。井号#用于注释，美元符号$用于变量，和号&用于位运算，竖线|用于或运算。问号?用于条件表达式，下划线_用于命名。这些符号是编程语言的基本组成部分。",
    
    # 问答类 (10句) - 每句约300-400字
    "什么是人工智能？人工智能是计算机科学的一个重要分支，它致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。人工智能的研究内容包括机器学习、计算机视觉、自然语言处理、专家系统、机器人学等多个领域。从1956年达特茅斯会议首次提出人工智能概念以来，这一领域经历了多次发展浪潮，如今已深入到我们生活的方方面面，从智能手机的语音助手到自动驾驶汽车，从医疗诊断系统到金融风控模型，人工智能正在改变着人类社会的运作方式。",
    
    "天空为什么是蓝色的？这是一个涉及物理学中光学原理的经典问题。当太阳光穿过大气层时，会与空气中的分子发生散射作用。太阳光由红橙黄绿青蓝紫七种颜色的光组成，其中蓝光的波长较短，更容易被大气分子散射到各个方向。这种现象被称为瑞利散射，由英国物理学家瑞利勋爵在19世纪末发现并解释。正是因为这种散射效应，当我们抬头仰望天空时，看到的是被散射的蓝光，所以天空呈现出美丽的蓝色。在日出和日落时分，阳光需要穿过更厚的大气层，蓝光被散射殆尽，留下的红橙光则形成了绚丽的晚霞。",
    
    "地球绕太阳转一圈需要多久？地球围绕太阳公转的周期大约是365.2422天，这就是我们所说的一个回归年。为了方便历法计算，我们通常将一年定为365天，并通过每四年增加一天的方式（即闰年）来修正这个微小的时间差。地球公转轨道是一个椭圆，太阳位于椭圆的一个焦点上。在公转过程中，地球距离太阳的距离会发生变化，近日点约在每年一月初，远日点约在七月初。地球公转的同时还在自转，自转周期约为24小时，这造就了昼夜更替的现象。地球公转轨道面与赤道面之间存在约23.5度的夹角，这个倾角是造成四季变化的主要原因。",
    
    "水的化学式是什么？水的化学式是H2O，这意味着每个水分子由两个氢原子和一个氧原子组成。水是地球上最常见的物质之一，覆盖了地球表面约71%的面积。水分子呈V形结构，氧原子与两个氢原子之间的夹角约为104.5度。由于氧原子的电负性比氢原子强，水分子具有极性，这使得水成为一种优良的溶剂，能够溶解许多物质，因此被称为万能溶剂。水在常温常压下呈液态，但在0摄氏度以下会凝固成冰，在100摄氏度以上会沸腾变成水蒸气。水的比热容很大，这意味着它能够吸收和储存大量的热量，对地球气候起着重要的调节作用。",
    
    "中国的首都在哪里？中国的首都是北京，这是一座拥有三千多年建城史和八百多年建都史的历史文化名城。北京位于华北平原的北部，背靠燕山，东临天津，其余方向与河北省相邻。作为中华人民共和国的政治中心、文化中心、国际交往中心和科技创新中心，北京汇聚了国家最高权力机关、中央政府各部门、外国驻华使馆等重要机构。北京拥有故宫、天坛、颐和园、长城等众多世界文化遗产，是世界上最受欢迎的旅游目的地之一。这座城市既保留着胡同四合院等传统风貌，又拥有鸟巢、水立方、大兴机场等现代建筑奇迹，古老与现代在这里完美交融。",
    
    "一年有几个季节？一年有春夏秋冬四个季节，这是地球公转和地轴倾斜共同作用的结果。春季通常从三月到五月，万物复苏，草木萌发，是播种希望的季节。夏季从六月到八月，阳光充足，气温升高，是生长旺盛的时节。秋季从九月到十一月，天高气爽，硕果累累，是收获的季节。冬季从十二月到次年二月，天寒地冻，万物休眠，是积蓄能量的时期。在中国传统文化中，二十四节气详细划分了一年中的气候变化，指导着农业生产和日常生活。春种夏长秋收冬藏，四季轮回蕴含着深刻的自然规律和人生哲理。",
    
    "人类需要呼吸什么气体？人类需要呼吸氧气来维持生命活动。氧气是空气的主要成分之一，约占空气体积的21%。当我们吸气时，氧气通过呼吸道进入肺部，穿过肺泡壁进入血液，与红细胞中的血红蛋白结合，被输送到全身各个组织细胞。在细胞内，氧气参与有氧呼吸过程，将葡萄糖等有机物氧化分解，释放出维持生命所需的能量。与此同时，产生的二氧化碳通过血液循环回到肺部，在呼气时被排出体外。成年人每分钟呼吸约12到20次，每天呼吸约两万次，吸入约一万升空气。氧气对人类如此重要，以至于缺氧几分钟就可能导致脑细胞不可逆的损伤。",
    
    "太阳从哪个方向升起？太阳从东方升起，这是地球自西向东自转造成的视觉效果。每天清晨，随着地球的自转，太阳逐渐出现在东方的地平线上，这就是日出。实际上，太阳本身并没有移动，而是地球在转动，让我们看到太阳从东方升起，经过南方天空，最后在西方落下。在北半球，正午时分太阳位于正南方；在南半球，正午时分太阳则位于正北方。每年春分和秋分这两天，太阳从正东方升起，正西方落下。夏至时，太阳升起的位置偏北，冬至时则偏南。古人通过观察日出方位的变化来制定历法、确定节气，指导农业生产。",
    
    "鱼儿为什么能在水里呼吸？鱼类能够在水中呼吸是因为它们拥有特殊的呼吸器官——鳃。鳃由许多细小的鳃丝组成，内部密布着毛细血管。当水流过鳃丝时，溶解在水中的氧气透过鳃丝上皮细胞进入血液，而血液中的二氧化碳则排入水中，完成气体交换。水的含氧量远低于空气，但鳃的特殊结构大大增加了气体交换的表面积，使鱼类能够从水中获得足够的氧气。不同鱼类的鳃结构有所差异，适应不同的生活环境。有些鱼类如肺鱼，除了鳃之外还进化出了类似肺的器官，能够在缺氧的水域中直接呼吸空气。鱼类的呼吸方式展示了生命对环境的奇妙适应。",
    
    "什么是友谊？友谊是人与人之间建立在相互理解、信任和关爱基础上的真挚情感。真正的友谊超越了利益交换，是一种纯粹的心灵连接。朋友之间能够分享快乐，分担忧愁，在困难时刻相互扶持，在成功时刻共同庆祝。友谊需要用心经营，需要真诚的沟通、包容的心态和持续的付出。古人云：君子之交淡如水，意思是真正的友谊不需要刻意讨好，而是自然真诚的相处。在人生旅途中，能够遇到知心朋友是一种幸运，他们像镜子一样帮助我们认识自己，像灯塔一样指引我们前行。友谊不分年龄、性别、地位，是人与人之间最美好的情感纽带之一。",
    
    # 对话类 (10句) - 每句约300-400字
    "你好，很高兴认识你。你好，我也很高兴认识你。这是我们第一次见面时最常用的问候语。在社交场合中，初次见面的问候往往决定了后续交流的氛围。真诚的微笑、适当的眼神接触、得体的握手或点头致意，都能传达出友好和尊重的态度。认识新朋友是人生中美好的体验，每一次相遇都可能开启一段新的故事。有些人可能只是匆匆过客，有些人却可能成为一生的挚友。保持开放和真诚的心态，主动与他人交流，是拓展社交圈、丰富人生阅历的重要方式。在这个快节奏的现代社会，我们更应该珍惜每一次真诚的相遇。",
    
    "今天天气怎么样？今天天气晴朗，适合外出。天气是人们日常交流中最常见的话题之一，因为它直接影响着我们的出行安排和活动计划。晴朗的天气意味着阳光明媚、万里无云，是户外活动的绝佳时机。在这样的日子里，可以去公园散步、去郊外踏青、去运动场锻炼，或者只是简单地坐在阳光下享受温暖。天气预报技术的发展让我们能够提前了解天气变化，做好相应的准备。不过，天气的变化有时也充满惊喜，一场突如其来的春雨、一道绚丽的彩虹，都可能成为一天中最美好的记忆。关注天气，也是在关注自然、关注生活的细节。",
    
    "你喜欢什么颜色？我喜欢蓝色，因为它像大海。颜色是视觉世界中最直观的元素，每个人都有自己偏爱的颜色，这往往与个人性格、经历和情感有关。蓝色是一种让人感到宁静和宽广的颜色，它让人联想到深邃的大海、辽阔的天空。蓝色分为许多不同的色调，从深沉的藏蓝到明亮的天蓝，每一种都有独特的魅力。心理学研究表明，蓝色能够让人心情平静，有助于思考和专注。喜欢蓝色的人通常性格沉稳、理性、富有想象力。颜色不仅是视觉的享受，也是情感的表达，通过颜色我们可以更好地了解自己和他人。",
    
    "你喜欢吃什么水果？我最喜欢吃苹果和香蕉。水果是大自然赐予人类的美味礼物，不仅口感鲜美，还富含维生素、矿物质和膳食纤维，对健康大有裨益。苹果被称为水果之王，有红富士、青苹果、黄元帅等多个品种，口感从脆甜到酸甜各有特色。苹果富含果胶和抗氧化物质，有句谚语说：一天一苹果，医生远离我。香蕉则是热带水果的代表，软糯香甜，富含钾元素，是运动员和健身爱好者的理想零食。不同季节有不同的时令水果，春天的草莓、夏天的西瓜、秋天的葡萄、冬天的柑橘，每个季节都有独特的水果可以品尝。",
    
    "你周末通常做什么？我通常看书或散步。周末是工作之余难得的休息时光，每个人度过周末的方式各不相同。阅读是一种能够丰富心灵的活动，通过书籍我们可以穿越时空，与古今中外的智者对话，拓宽视野、增长见识。无论是小说、散文、历史还是科普，每一本书都是一扇通往新世界的门。散步则是一种简单而有效的放松方式，漫步在公园、街道或乡间小路上，呼吸新鲜空气，观察周围的人事物，让身心从忙碌中解脱出来。有些人喜欢在周末聚会、运动、旅行，有些人则喜欢宅在家里休息。无论怎样度过，周末的意义在于让生活张弛有度，为下一周积蓄能量。",
    
    "你会说几种语言？我会说中文和一点英语。语言是人类交流思想、传递文化的重要工具。世界上有数千种语言，每一种都承载着独特的文化内涵。中文是世界上使用人数最多的语言，有着悠久的历史和丰富的表达方式，汉字更是世界上唯一仍在使用的表意文字系统。英语则是使用范围最广的语言，是国际交流、商业贸易、科学研究中的通用语言。学习一门新语言不仅是掌握一种沟通技能，更是打开一扇了解不同文化的窗户。在全球化时代，掌握多种语言的人拥有更广阔的发展空间。语言学习需要长期的积累和练习，但每学会一个新词汇、每能表达一个新意思，都是值得庆祝的进步。",
    
    "你家有几口人？我家有四口人。家庭是社会的细胞，是每个人成长的港湾。四口之家通常包括父母和两个孩子，这是一种常见的家庭结构。在中国传统文化中，家庭观念非常重要，尊老爱幼、和睦相处是家庭美德的核心。每个家庭成员都有自己的角色和责任，父母负责养育和引导，孩子负责学习和成长。家庭生活中充满了日常的琐碎和温馨的时刻：一起吃饭、一起看电视、一起过节、一起面对困难。随着社会的发展，家庭结构也在变化，有核心家庭、三代同堂、单亲家庭等多种形式。无论形式如何，家庭成员之间的爱与支持始终是家庭最珍贵的财富。",
    
    "你最喜欢的运动是什么？我最喜欢打篮球。运动是保持身心健康的重要方式，每个人都可以找到适合自己的运动项目。篮球是一项充满激情和团队精神的运动，它不仅锻炼身体素质，还培养协作能力和竞争意识。在篮球场上，运球、传球、投篮、防守，每一个动作都需要技巧和配合。一场篮球比赛就像一场精彩的舞蹈，需要队员之间默契的配合、灵活的战术变化。篮球运动起源于美国，如今已风靡全球，NBA更是成为世界顶级篮球联赛的代名词。无论是职业球员还是业余爱好者，都能在篮球运动中找到乐趣。运动不分年龄，从小培养运动习惯，能够受益终生。",
    
    "你觉得学习重要吗？是的，学习非常重要。学习是人类进步的阶梯，是个人成长的核心途径。从出生开始，人就在不断学习：学习走路、学习说话、学习认识世界。学校教育是系统学习知识的重要阶段，但学习并不止步于校园。在知识爆炸的时代，终身学习已经成为每个人的必修课。学习不仅限于书本知识，还包括生活技能、社交能力、情感管理等各个方面。通过学习，我们能够拓展视野、提升能力、改变命运。古人说学无止境，意思是学习没有终点，只有不断学习才能跟上时代的步伐。保持好奇心和求知欲，是持续学习的动力源泉。学习让人生更加丰富，让未来充满可能。",
    
    "你有什么爱好？我喜欢画画和听音乐。爱好是工作学习之外的调味剂，是个人兴趣的自然延伸。画画是一种视觉艺术，通过线条和色彩表达内心的想法和情感。无论是素描、水彩、油画还是数字绘画，每一种形式都有独特的魅力。画画的过程是专注的、安静的，能够让人忘却烦恼，沉浸在创作的世界中。音乐则是听觉的艺术，能够直接触动人心。不同类型的音乐带来不同的感受：古典音乐优雅深沉，流行音乐轻松活泼，民族音乐独具风情。听音乐可以放松心情、激发灵感、陪伴孤独。拥有爱好的人生活更加充实，因为爱好不仅是消遣，更是自我表达和心灵滋养的方式。",
    
    # 描述类 (10句) - 每句约300-400字
    "春天来了，花儿开放，小鸟在枝头歌唱。这是一年中最令人期待的季节，大地从沉睡中苏醒，万物开始展现新的生机。春风轻柔地吹过田野，带来泥土的芬芳和花草的清香。柳树抽出了嫩绿的新芽，在河边摇曳生姿。桃花、杏花、梨花竞相开放，粉的如霞、白的似雪，装点着山野和庭院。燕子从南方飞回，在屋檐下筑巢，叽叽喳喳地唱着春天的歌。农民们开始忙碌起来，翻耕土地、播种希望。孩子们脱去厚重的冬装，在田野上奔跑放风筝。春天的阳光温暖而不炽热，春天的雨水细密而滋润，一切都充满着新生的力量和美好的希望。",
    
    "夏天的太阳很热，人们喜欢去海边游泳。夏季是一年中最炎热的季节，烈日当空，大地蒸腾着热气。知了在树上不停地鸣叫，仿佛在抱怨天气的炎热。城市里柏油马路被晒得发软，行人们撑着遮阳伞匆匆赶路。空调房成了最受欢迎的地方，冰镇西瓜和冷饮是消暑的最佳选择。不过，夏天也有独特的魅力：茂密的树冠投下大片阴凉，荷花在池塘中亭亭玉立，夜晚的萤火虫闪烁着梦幻的光芒。海边是夏天最热闹的地方，金色的沙滩、碧蓝的海水、翻滚的浪花，人们在海水中嬉戏，享受清凉。夏天的夜晚，繁星满天，蛙声一片，别有一番风味。",
    
    "秋天树叶变黄，纷纷飘落在地上。秋季是收获的季节，也是一年中最富有诗意的时节。当第一阵秋风吹过，树叶开始变换颜色，从翠绿到金黄、从金黄到火红，层林尽染，美不胜收。银杏叶像一把把金色的小扇子，枫叶像一团团燃烧的火焰，在阳光下熠熠生辉。落叶随风飘舞，铺满小径，踩上去沙沙作响。田野里稻谷金黄，果园里硕果累累，农民们忙着收割，脸上洋溢着丰收的喜悦。秋天的天空格外高远，云淡风轻，让人心旷神怡。菊花在秋风中绽放，桂花散发着浓郁的香气。秋天让人感受到生命的成熟和岁月的沉淀，是一个值得细细品味的季节。",
    
    "冬天寒冷，人们穿上厚厚的棉衣。冬季是一年中最寒冷的季节，北风呼啸，天寒地冻。清晨的窗户上结满了美丽的冰花，屋檐下挂着晶莹的冰凌。人们穿上羽绒服、戴上帽子和手套，全副武装地抵御严寒。河流湖泊结了厚厚的冰，成了天然的滑冰场。当雪花纷纷扬扬地飘落，整个世界变成了银装素裹的童话王国。孩子们最开心，堆雪人、打雪仗，在雪地里留下欢快的笑声。冬天虽然寒冷，却也有独特的温暖：一家人围坐在火炉旁，喝着热腾腾的汤，聊着家常，那份温馨足以驱散所有的寒意。冬至吃饺子、春节团圆，冬天里藏着最浓的亲情和年味。",
    
    "早晨的阳光温暖而柔和，照亮了大地。清晨是一天中最清新的时刻，当第一缕阳光穿透薄雾，洒向大地，世界从沉睡中慢慢醒来。东方的天际先是泛起鱼肚白，然后渐渐染上红霞，最后太阳露出笑脸，金色的光芒向四面八方伸展。草叶上挂着的露珠在阳光下闪闪发光，像一颗颗晶莹的珍珠。鸟儿开始了晨间的合唱，此起彼伏的鸣叫声唤醒了沉睡的世界。街道上晨练的人们或跑步或打太极，公园里弥漫着清新的空气。早餐摊上冒着热气，油条的香味、豆浆的醇香飘散开来。早晨的阳光不似正午那般炽烈，带着一种温柔的暖意，给人希望和力量，预示着美好一天的开始。",
    
    "夜晚的星空美丽，无数星星闪烁着光芒。当夜幕降临，白日的喧嚣渐渐退去，天空变成了一个巨大的舞台。在没有光污染的地方，抬头就能看见满天繁星，密密麻麻地镶嵌在深蓝色的天幕上。最亮的那颗是金星，还有北斗七星指引方向，银河像一条银色的河流横贯天际。古人通过观测星空制定历法、导航定位，还编织了许多美丽的神话故事：牛郎织女隔河相望，嫦娥奔月独守广寒。现代天文学家通过望远镜探索宇宙的奥秘，发现了无数的星系、星云和黑洞。仰望星空，让人感受到宇宙的浩瀚和人类的渺小，也激发起探索未知的渴望。星空下许下的愿望，承载着人们对美好未来的憧憬。",
    
    "大海辽阔无边，波浪拍打着沙滩。海洋占据了地球表面的大部分面积，是地球上最广阔的水域。站在海边眺望，只见海天一色，无边无际，让人感受到大自然的壮阔与雄伟。海浪一浪接一浪地涌来，拍打着沙滩和礁石，发出轰鸣的声响，激起白色的浪花。潮起潮落，周而复始，这是月球引力造成的自然现象。海水中生活着无数种生物，从微小的浮游生物到巨大的鲸鱼，构成了复杂而精彩的海洋生态系统。海滨是人们休闲度假的好去处，阳光、沙滩、海浪、椰林，构成热带风情的美丽画卷。大海也是人类探索的重要领域，海底蕴藏着丰富的资源，深海中还有许多未知的奥秘等待发现。",
    
    "高山巍峨壮观，山顶常年覆盖着白雪。山脉是地球上最雄伟的地貌之一，高耸入云，气势磅礴。世界上最高的山峰是珠穆朗玛峰，海拔8848米，被称为世界屋脊。在高海拔地区，气温随高度增加而降低，所以山顶常年积雪，即使夏日也不融化。登山是一项充满挑战的运动，攀登者需要克服缺氧、严寒、陡峭等重重困难，才能到达顶峰。站在山顶俯瞰，云海翻腾，群山连绵，那种征服自然的成就感无与伦比。山脉也是重要的水源地，积雪融化形成河流，滋养着山下的平原和城市。山中往往有丰富的动植物资源，从山脚到山顶，植被随海拔变化呈现出明显的垂直分布。高山既是自然的奇观，也是人类挑战极限的舞台。",
    
    "森林里树木茂密，各种动物在其中生活。森林是地球的肺，是陆地上最重要的生态系统之一。走进森林，只见参天大树遮天蔽日，阳光从树叶的缝隙中洒落，形成斑驳的光影。林间弥漫着清新的空气，富含负氧离子，让人感到神清气爽。森林中生活着各种各样的动物：松鼠在树枝间跳跃，鹿在林间空地上吃草，鸟儿在树梢上筑巢歌唱，昆虫在草丛中忙碌。森林的层次分明，从高大的乔木到低矮的灌木再到地面的草本植物，每一层都有独特的生物群落。森林还具有重要的生态功能：涵养水源、保持水土、调节气候、净化空气。保护森林就是保护地球的生态平衡，就是保护人类自己的家园。",
    
    "城市里高楼林立，街道上人来人往。城市是人类文明的重要标志，是现代生活的主要舞台。走进大城市，首先映入眼帘的是鳞次栉比的高楼大厦，玻璃幕墙在阳光下闪闪发光。摩天大楼直插云霄，写字楼、商场、住宅楼构成了城市的骨架。街道上车水马龙，行人熙熙攘攘，商店橱窗琳琅满目，霓虹灯闪烁不息。城市是经济活动的中心，工厂、公司、银行、交易所汇聚于此，创造着巨大的财富。城市也是文化的中心，学校、图书馆、博物馆、剧院丰富了人们的精神生活。城市生活节奏快、机会多、便利性高，但也面临着交通拥堵、环境污染、住房紧张等问题。城市规划和管理的重要性日益凸显，建设宜居城市是现代城市发展的重要目标。",
    
    # 推理类 (10句) - 每句约300-400字
    "如果下雨，地面会变湿。现在地面湿了，所以下雨了。这是一个经典的逻辑推理例子，展示了因果关系的推理过程。然而，仔细分析就会发现这个推理存在问题：地面变湿确实可能是下雨导致的，但也可能是其他原因，比如洒水车经过、水管漏水、有人泼水等。正确的推理应该是：如果下雨，地面会变湿；现在下雨了，所以地面会变湿。这是从原因推导结果的演绎推理，是必然成立的。而反过来从结果推导原因，则属于归纳推理，只能得出可能性的结论。逻辑推理是人类思维的基本方式，帮助我们认识世界、做出判断。理解推理的正确形式和可能的谬误，是培养批判性思维的重要内容。",
    
    "所有的人都会死。苏格拉底是人，所以苏格拉底会死。这是古希腊哲学家最著名的三段论推理，是演绎推理的经典范例。三段论由大前提、小前提和结论三部分组成：大前提是普遍性的命题，小前提是特殊性的命题，结论是从大前提和小前提中必然推出的结果。在这个例子中，大前提是所有的人都会死，这是一个普遍真理；小前提是苏格拉底是人，这是一个具体事实；结论是苏格拉底会死，这是必然成立的。三段论是形式逻辑的核心内容，由亚里士多德系统提出，至今仍是逻辑学教学的基础。掌握三段论推理，能够帮助我们进行正确的思维，避免逻辑错误，做出合理的论证。",
    
    "学习使人进步。他学习很努力，所以他进步很快。这个推理展示了从一般原则到具体结论的演绎过程。学习是获取知识、提升能力的过程，通过学习，人们可以拓展视野、增长见识、掌握技能，从而在各个方面取得进步。这是一个普遍的规律，适用于所有人。当一个人学习很努力时，他投入了大量的时间和精力，认真钻研、勤于实践，自然会比不努力的人进步更快。当然，进步的速度还受到学习方法、天赋基础、环境条件等因素的影响。努力是进步的必要条件，但不是充分条件。理解这一点，可以帮助我们更全面地看待学习与进步的关系，既重视努力，也注重方法和效率。",
    
    "运动有益健康。她每天运动，所以她很健康。这个推理将一般的健康原则应用到具体的人身上。运动对健康的益处已经被大量科学研究所证实：运动能够增强心肺功能、提高免疫力、控制体重、改善情绪、预防慢性疾病。世界卫生组织建议成年人每周至少进行150分钟的中等强度有氧运动。当一个人每天坚持运动时，她就在持续地获得这些健康益处，身体素质自然会比较好。不过，健康是一个综合性的状态，除了运动之外，还与饮食、睡眠、心理状态、遗传因素等有关。运动是健康的重要保障，但健康还需要其他方面的配合。这个推理提醒我们养成运动习惯的重要性，同时也让我们理解健康的多元因素。",
    
    "读书可以增长知识。他读了很多书，所以知识丰富。这个推理展示了阅读与知识积累之间的关系。书籍是人类文明的结晶，记录了前人的智慧、经验和发现。通过阅读，我们可以跨越时空的限制，与古今中外的思想家、科学家、文学家对话，汲取他们的知识精华。一个人读了很多书，意味着他接触了广泛的信息和观点，在各个领域都有所涉猎，自然比不读书的人知识更加丰富。当然，知识的丰富程度不仅取决于读书的数量，还取决于阅读的质量、理解的深度、思考的能力。单纯地浏览而不思考，可能只是信息的堆积而非真正的知识。这个推理鼓励我们多读书，同时也提醒我们要读好书、深思考。",
    
    "勤奋是成功的关键。他很勤奋，所以他成功了。这个推理将勤奋这一品质与成功这一结果联系起来。勤奋意味着努力工作、不懈奋斗、持之以恒，这些品质确实是取得成功的重要因素。许多成功人士的故事都证明了这一点：爱迪生经过上千次实验才发明了电灯，居里夫人在简陋的实验室里坚持研究最终发现镭元素，运动员们日复一日地训练才能在赛场上夺冠。当一个人很勤奋时，他付出了比常人更多的努力，克服了更多的困难，积累了更多的经验和能力，成功的机会自然更大。不过，成功还需要方向正确、机遇合适、能力匹配等条件。勤奋是成功的必要条件，但不是唯一条件。这个推理肯定了勤奋的价值，也让我们理解成功的复杂性。",
    
    "节约用水很重要。我们要养成节约用水的好习惯。这个推理从价值判断过渡到行动建议。水是生命之源，是人类社会生存和发展不可或缺的资源。然而，地球上的淡水资源非常有限，只占总水量的不到3%，而且分布极不均匀。随着人口增长和经济发展，水资源短缺已经成为全球性问题。节约用水不仅是个人美德的体现，更是对社会责任的担当。养成节约用水的好习惯，可以从日常小事做起：洗菜水可以用来浇花，洗澡时间可以适当缩短，漏水的水龙头要及时修理。这些看似微不足道的举动，汇聚起来就是巨大的节约。节约用水是对自然的尊重，是对后代的负责，是可持续发展的重要实践。",
    
    "保护环境人人有责。每个人都应该爱护环境。这个推理强调了环境保护的主体普遍性。环境是人类赖以生存的基础，空气、水、土壤、森林、海洋，这些自然资源维系着人类的生存和发展。然而，工业化进程带来了严重的环境问题：大气污染、水体污染、土壤退化、生物多样性减少、气候变化。环境保护不是某一个国家、某一个组织或某一部分人的责任，而是生活在这个地球上的每一个人的共同责任。每个人都可以为环境保护做出贡献：减少使用一次性塑料制品、垃圾分类投放、选择公共交通出行、支持环保产品。个人的力量虽然有限，但当每个人都行动起来，就能形成强大的合力。爱护环境就是爱护我们共同的家园。",
    
    "诚实是一种美德。我们要做一个诚实的人。这个推理从道德判断引出行为准则。诚实是中华民族的传统美德之一，意味着说话真实、做事坦荡、不欺骗、不作假。诚实的人赢得他人的信任和尊重，建立良好的人际关系和社会声誉。相反，谎言和欺骗可能带来短期利益，但最终会失去信任，陷入孤立。在社会层面，诚实是信用体系的基础，是市场经济正常运转的前提。做一个诚实的人，需要在日常生活中坚持说真话、办实事、守承诺，即使面临诱惑或压力也不动摇。诚实不仅是一种行为选择，更是一种人格修养。这个推理引导我们认识诚实的价值，并在生活中践行这一美德。",
    
    "友谊需要珍惜。好朋友之间要互相信任和帮助。这个推理从友谊的价值出发，提出了维护友谊的具体要求。友谊是人生中最珍贵的情感之一，真正的朋友能够分享快乐、分担痛苦、相互支持、共同成长。然而，友谊不是理所当然的，它需要双方共同的付出和维护。珍惜友谊意味着重视这份感情，不因忙碌而忽视，不因小事而伤害。信任是友谊的基石，朋友之间要坦诚相待、信守承诺、不背叛、不猜疑。帮助是友谊的体现，朋友有难时要伸出援手，朋友成功时要真心祝福。真正的友谊经得起时间的考验，经得起风雨的洗礼。这个推理提醒我们，拥有友谊是幸运，维护友谊需要用心，珍惜友谊才能长久。",
    
    # 情感类 (10句) - 每句约300-400字
    "看到家人团聚，我感到非常幸福和温暖。家是每个人心中最柔软的地方，家人团聚是世间最温馨的画面。无论走得多远，无论经历了什么，回到家中看到亲人的笑脸，所有的疲惫和烦恼都会烟消云散。团聚的时刻，大家围坐在一起，吃着家常菜，聊着各自的生活，孩子们的笑声、长辈们的叮咛、兄弟姐妹间的打趣，构成了一幅和谐美满的画面。这种幸福不是物质能够衡量的，它源于血脉相连的亲情，源于相互关爱的温暖。在快节奏的现代社会，家人团聚的机会越来越少，因此每一次团聚都显得格外珍贵。珍惜与家人在一起的时光，就是珍惜生命中最真挚的情感。",
    
    "听到好消息，她激动得跳了起来。喜悦是人类最美好的情感之一，当期待已久的好消息终于到来，那种激动和兴奋难以自抑。也许是通过了重要的考试，也许是获得了理想的工作，也许是久别重逢的亲人即将归来，无论是什么好消息，都会让人心跳加速、热泪盈眶。她跳了起来，这是喜悦之情最自然的身体表达，是内心激动的外在流露。在那一刻，所有的付出和等待都得到了回报，所有的焦虑和担忧都化作了欢欣。喜悦是可以传递的，她的激动也会感染身边的人，让大家一起分享这份快乐。人生中这样的高光时刻或许不多，但每一次都值得铭记和珍藏。",
    
    "失去宠物后，他伤心了好几天。宠物是人类忠实的伙伴，它们用无条件的爱和陪伴温暖着主人的生活。当一只朝夕相处的宠物离世，那种失去的痛苦是真实而深刻的。他伤心了好几天，吃不下饭、睡不着觉，脑海中不断浮现宠物生前的画面：它迎接自己回家时的欢快、它蜷缩在脚边睡觉时的安详、它用头蹭自己手心时的亲昵。这种悲伤不是矫情，而是对一份真挚情感的悼念。时间会慢慢抚平伤口，但那份陪伴的记忆永远不会消失。失去宠物让人学会珍惜，珍惜身边的每一个生命，珍惜当下的每一份陪伴。悲伤是爱的延续，是生命对生命的敬意。",
    
    "考试取得好成绩，同学们都很开心。考试是学生时代最重要的评价方式之一，好成绩是对努力的肯定，是对能力的证明。当考试结果公布，看到自己的名字排在前列，那种喜悦和自豪油然而生。同学们都很开心，不仅为自己，也为彼此的进步感到高兴。这份开心背后是无数个日夜的付出：课堂上专注的听讲、作业时认真的思考、复习时反复的练习。好成绩不是偶然的，它是汗水和智慧的结晶。当然，成绩只是评价的一个维度，不是衡量一个人价值的唯一标准。但取得好成绩的喜悦是真实的，它让人相信付出终有回报，让人对未来充满信心。这份喜悦会成为继续前行的动力。",
    
    "收到朋友的礼物，她感到很惊喜。礼物是友谊的物质载体，承载着送礼人的心意和祝福。当一份精心准备的礼物出现在面前，那种惊喜的感觉是难以言喻的。也许是一本书，正合她的阅读兴趣；也许是一件饰品，恰好搭配她的衣服；也许只是一张手写的卡片，字里行间却满是真挚的情谊。惊喜源于意外，源于被惦记的感动。朋友在众多选择中想到了她，花时间挑选、包装、寄送，这份用心比礼物本身更加珍贵。她感到惊喜，不仅因为收到了礼物，更因为感受到了友谊的温度。礼物有价，情谊无价，这份惊喜会成为她记忆中温暖的片段。",
    
    "看到美丽的风景，心情变得格外舒畅。大自然是最伟大的艺术家，山川湖海、日出日落、花开叶落，每一处风景都蕴含着独特的美。当我们置身于美丽的风景之中，眼前的壮阔或秀丽会让人忘却烦恼，心情变得格外舒畅。站在山顶俯瞰云海，面对大海聆听涛声，漫步森林呼吸清新，每一个场景都有治愈心灵的力量。风景之美不仅在于视觉的享受，更在于心灵的触动。它让人感受到自然的伟大和个人的渺小，让人放下执念，获得平静。现代生活的压力常常让人喘不过气，走进大自然、欣赏美丽风景，是舒缓心情、恢复能量的最佳方式。风景是自然的馈赠，心情舒畅是对这份馈赠的回应。",
    
    "帮助别人让我感到快乐和满足。助人为乐是一种高尚的品质，也是一种独特的幸福来源。当我们向他人伸出援手，解决了一个困难、带来了一丝温暖，内心会涌起一种难以言喻的快乐和满足。这种快乐不同于物质享受带来的愉悦，它源于自我价值的实现，源于与他人情感的连接。帮助别人的形式多种多样：为迷路者指引方向、为老人让座、为同学解答问题、为社区做志愿服务。每一次帮助都是善意的传递，每一次付出都是爱的播种。当我们帮助别人时，也在丰富自己的内心，提升自己的人格。快乐和满足是善良的回响，是人性光辉的体现。",
    
    "回忆童年时光，心中充满温馨和怀念。童年是人生中最纯真、最无忧无虑的时光，当长大成人后回首往事，那些童年的记忆总是带着温暖的光芒。也许是和小伙伴们在田野里追逐嬉戏，也许是在奶奶怀里听古老的故事，也许是过年时穿新衣放鞭炮的欢喜，也许是得到一颗糖果时的小小满足。那时的世界简单而美好，快乐来得容易，悲伤去得也快。童年时光一去不复返，但那些记忆永远珍藏在心底。温馨是因为曾经拥有，怀念是因为无法重来。回忆童年，是对逝去时光的致敬，也是对纯真年代的眷恋。那份温馨和怀念，是成长路上最柔软的慰藉。",
    
    "面对困难时，我们要保持乐观的心态。人生不如意事十之八九，困难和挫折是每个人都会遇到的挑战。面对困难，悲观的人可能感到绝望和放弃，乐观的人却能看到希望和转机。乐观的心态不是盲目的自信，而是在困境中依然相信问题可以解决、未来可以更好。这种心态让人保持冷静和理智，积极寻找解决问题的方法，而不是沉浸在消极情绪中。研究表明，乐观的人更容易克服困难、实现目标、获得幸福。保持乐观并不意味着忽视问题的存在，而是选择以积极的态度去面对。困难是人生的必修课，乐观是应对困难的最佳武器。心态决定状态，乐观引领前行。",
    
    "成功后的喜悦让人难以忘怀。成功是对努力的回报，是对梦想的实现。当经过长期的奋斗终于达成目标，那种成功的喜悦是人生中最美妙的体验之一。也许是创业成功后的激动，也许是作品发表后的欣慰，也许是比赛夺冠后的狂喜，每一种成功都有独特的喜悦滋味。这种喜悦让人难以忘怀，因为它凝聚了太多的付出和期待。成功的那一刻，所有的艰辛都变成了值得，所有的质疑都变成了动力。成功后的喜悦不仅是对过去的肯定，更是对未来的激励。它让人相信梦想可以实现，让人有勇气追求更高的目标。这份难以忘怀的喜悦，会成为人生旅途中最闪亮的坐标。",
    
    # 代码类 (10句) - 测试代码上下文因果关系，每句约300-400字
    "定义一个变量x等于10，后面使用x进行计算。代码示例：x = 10; y = x + 5; print(y) 输出15。变量是程序中最基本的概念，用于存储数据。在这段代码中，首先定义变量x并赋值为10，然后在下一行使用x进行计算，将x加5的结果赋给y。这展示了变量的定义和使用的因果关系：先定义后使用。如果在使用前没有定义变量，程序会报错。变量的值可以被修改，例如执行x = x + 1后，x的值变为11。理解变量的定义和使用是学习编程的第一步。",
    
    "函数add接收两个参数a和b，返回它们的和。代码示例：def add(a, b): return a + b; result = add(3, 5); print(result) 输出8。函数是组织代码的基本单元，用于封装可重用的逻辑。在这段代码中，首先定义函数add，然后调用它计算3加5。定义和调用之间存在因果关系：先定义后调用。函数可以被多次调用，每次传入不同的参数。例如add(10, 20)返回30。良好的函数设计使代码更加模块化、可读、可维护。",
    
    "如果x大于0，执行第一个分支，否则执行第二个分支。代码示例：x = 5; if x > 0: print('正数') else: print('非正数') 输出正数。条件语句根据条件决定执行哪段代码。在这段代码中，x等于5大于0为真，所以执行第一个分支打印正数。如果将x改为负数如-3，则会执行else分支打印非正数。条件语句实现了程序的分支逻辑，使得程序能够根据不同情况做出不同的响应。",
    
    "for循环从0遍历到9，每次迭代打印当前数字。代码示例：for i in range(10): print(i) 输出0到9。循环语句用于重复执行某段代码。在这段代码中，range(10)生成0到9的数字序列，for循环依次将每个数字赋给变量i并打印。循环会执行10次。循环可以嵌套，例如双重循环遍历二维数组。循环控制语句break可以提前退出循环，continue可以跳过当前迭代。",
    
    "定义类Person，包含属性name和age，以及方法introduce。代码示例：class Person: def __init__(self, name, age): self.name = name; self.age = age; def introduce(self): print(f'我是{self.name}，{self.age}岁'); p = Person('张三', 25); p.introduce() 输出我是张三，25岁。类是面向对象编程的核心概念。在这段代码中，首先定义类Person，然后创建实例p，最后调用方法introduce。类定义和实例化之间存在因果关系：先定义类才能创建实例。",
    
    "导入numpy库，使用np.array创建数组。代码示例：import numpy as np; arr = np.array([1, 2, 3, 4, 5]); print(arr.sum()) 输出15。库是预先编写好的代码集合。在这段代码中，首先导入numpy库，然后使用np.array创建数组，最后调用sum方法计算总和。导入和使用之间存在因果关系：先导入才能使用。numpy是Python中最常用的科学计算库。",
    
    "打开文件data.txt，读取内容并关闭文件。代码示例：f = open('data.txt', 'r'); content = f.read(); f.close(); print(content)。文件操作是程序与外部存储交互的基本方式。在这段代码中，首先打开文件，然后读取内容，最后关闭文件。这三个操作有严格的顺序关系：必须先打开才能读取，读取完成后应该关闭。使用with语句可以自动管理文件关闭：with open('data.txt') as f: content = f.read()。",
    
    "定义列表numbers等于1,2,3,4,5，使用索引访问第三个元素。代码示例：numbers = [1, 2, 3, 4, 5]; print(numbers[2]) 输出3。列表是Python中最常用的数据结构。在这段代码中，首先定义列表，然后使用索引2访问第三个元素（索引从0开始）。列表支持切片操作，如numbers[1:3]返回[2, 3]。列表是可变的，可以添加、删除、修改元素。",
    
    "try语句捕获除零错误，打印错误信息。代码示例：try: result = 10 / 0; except ZeroDivisionError: print('除零错误') 输出除零错误。异常处理是程序应对运行时错误的机制。在这段代码中，try块尝试执行除零操作，这会引发ZeroDivisionError异常，except块捕获异常并打印错误信息。异常处理使程序能够优雅地处理错误，而不是直接崩溃。",
    
    "定义字典person等于name冒号张三, age冒号25，通过键访问值。代码示例：person = {'name': '张三', 'age': 25}; print(person['name']) 输出张三。字典是Python中存储键值对的数据结构。在这段代码中，首先定义字典，然后通过键name访问对应的值。字典的值通过键访问，如person['age']返回25。字典在表示结构化数据方面有广泛应用。",
    
    # 数学计算类 (10句) - 测试数学推理能力，每句约300-400字
    "计算1加2等于3，这是最基本的加法运算。算式：1 + 2 = 3。加法是四则运算中最基础的一种，表示将两个或多个数合并成一个总数。在这道算式中，1和2是加数，3是和。加法满足交换律，即1加2等于2加1，结果都是3。加法也满足结合律，即(1加2)加3等于1加(2加3)，都等于6。加法可以扩展到多个数相加，如1加2加3加4等于10。在计算机中，加法是最基本的算术运算，由CPU的算术逻辑单元直接执行。",
    
    "计算10减3等于7，这是基本的减法运算。算式：10 - 3 = 7。减法是加法的逆运算，表示从一个数中减去另一个数，得到差值。在这道算式中，10是被减数，3是减数，7是差。减法不满足交换律，10减3等于7，但3减10等于负7。减法可以连续进行，如10减3减2等于5。减法的结果可以是负数，如3减10等于负7。减法在实际生活中应用广泛，如计算找零、计算剩余数量。",
    
    "计算4乘以5等于20，这是基本的乘法运算。算式：4 × 5 = 20。乘法表示相同数的重复相加，4乘以5表示5个4相加，即4加4加4加4加4等于20。在这道算式中，4和5是因数，20是积。乘法满足交换律，4乘以5等于5乘以4。乘法也满足结合律，(2乘以3)乘以4等于2乘以(3乘以4)，都等于24。乘法对加法满足分配律，2乘以(3加4)等于2乘以3加2乘以4，都等于14。",
    
    "计算20除以4等于5，这是基本的除法运算。算式：20 ÷ 4 = 5。除法是乘法的逆运算，表示将一个数分成若干等份。在这道算式中，20是被除数，4是除数，5是商。除法不满足交换律，20除以4等于5，但4除以20等于0.25。除法可能有余数，如7除以2等于3余1，写成7 ÷ 2 = 3...1。除数不能为零，除以零在数学上是无意义的。",
    
    "计算2的3次方等于8，这是乘方运算。算式：2³ = 8。乘方表示一个数自乘若干次，2的3次方表示3个2相乘，即2乘以2乘以2等于8。在这道算式中，2是底数，3是指数，8是幂。乘方运算有特定的运算规则：a的m次方乘以a的n次方等于a的(m加n)次方，如2的2次方乘以2的3次方等于2的5次方等于32。负指数表示倒数，如2的负1次方等于二分之一。",
    
    "计算根号9等于3，这是开方运算。算式：√9 = 3。开方是乘方的逆运算，根号9表示求哪个数平方后等于9，答案是3。在这道算式中，9是被开方数，3是平方根。正数有两个平方根，一正一负，如9的平方根是正3和负3，写成±3。开方运算可以推广到更高次方根，如立方根³√8等于2，因为2的3次方等于8。",
    
    "解方程x加2等于5，得到x等于3。方程：x + 2 = 5，解：x = 3。方程是含有未知数的等式，解方程就是求出未知数的值。在这道方程中，x是未知数，需要求出使等式成立的x的值。通过移项，将2移到等号右边变成负2，得到x等于5减2等于3。验证：将x等于3代入原方程，3加2等于5，等式成立，所以x等于3是正确答案。",
    
    "计算三角形面积等于底乘以高除以2。公式：S = (b × h) / 2。设三角形的底为b，高为h，则面积S等于b乘以h除以2。例如，底为6，高为4的三角形，面积等于6乘以4除以2等于12。验证：两个相同的三角形可以拼成一个平行四边形，平行四边形的面积是底乘以高，所以一个三角形的面积是它的一半。",
    
    "计算圆的面积等于π乘以半径的平方。公式：S = πr²。设圆的半径为r，则面积S等于π乘以r的平方，其中π约等于3.14159。例如，半径为5的圆，面积等于π乘以25，约等于78.54。圆的周长公式是C等于2乘以π乘以r，半径为5的圆周长约等于31.42。",
    
    "计算一元二次方程的解，使用求根公式。方程：ax² + bx + c = 0，求根公式：x = (-b ± √(b²-4ac)) / 2a。判别式Δ等于b的平方减4ac决定了根的情况：当Δ大于0时有两个不相等的实数根，当Δ等于0时有两个相等的实数根，当Δ小于0时没有实数根。例如，解方程x的平方减5x加6等于0，代入公式：a等于1，b等于负5，c等于6，Δ等于25减24等于1，x等于(5加减1)除以2，得到x等于3或x等于2。验证：3的平方减15加6等于0，2的平方减10加6等于0，答案正确。",
    
    # 编程代码类 (10句) - 测试代码理解能力，每句约300-400字
    "编写一个计算斐波那契数列的函数。代码：def fib(n): if n <= 1: return n; return fib(n-1) + fib(n-2)。斐波那契数列定义为F(0)等于0，F(1)等于1，F(n)等于F(n减1)加F(n减2)。调用fib(5)返回5，因为F(5)等于F(4)加F(3)等于3加2等于5。递归方法简洁但效率低，因为存在大量重复计算。更高效的方法是动态规划：def fib_dp(n): dp = [0, 1]; for i in range(2, n+1): dp.append(dp[i-1] + dp[i-2]); return dp[n]。斐波那契数列是理解递归和动态规划的经典案例。",
    
    "实现二分查找算法，在有序数组中查找目标值。代码：def binary_search(arr, target): left, right = 0, len(arr)-1; while left <= right: mid = (left + right) // 2; if arr[mid] == target: return mid; elif arr[mid] < target: left = mid + 1; else: right = mid - 1; return -1。二分查找时间复杂度为O(log n)。例如在数组[1,3,5,7,9]中查找7，第一次mid等于2，arr[2]等于5小于7，left变为3；第二次mid等于3，arr[3]等于7等于目标，返回3。二分查找要求数组必须是有序的。",
    
    "编写快速排序算法对数组进行排序。代码：def quicksort(arr): if len(arr) <= 1: return arr; pivot = arr[len(arr)//2]; left = [x for x in arr if x < pivot]; middle = [x for x in arr if x == pivot]; right = [x for x in arr if x > pivot]; return quicksort(left) + middle + quicksort(right)。快速排序平均时间复杂度为O(n log n)。例如排序[3,6,8,10,1,2,1]，选取pivot等于8，left等于[3,6,1,2,1]，middle等于[8]，right等于[10]，递归排序后得到[1,1,2,3,6,8,10]。",
    
    "实现深度优先搜索遍历图。代码：def dfs(graph, start, visited=None): if visited is None: visited = set(); visited.add(start); print(start); for neighbor in graph[start]: if neighbor not in visited: dfs(graph, neighbor, visited); return visited。DFS沿着一条路径尽可能深入。例如图graph等于{0:[1,2], 1:[2], 2:[0,3], 3:[]}，从节点0开始DFS，访问顺序可能是0、1、2、3。DFS可以用于检测环、拓扑排序、寻找路径。",
    
    "编写广度优先搜索遍历图。代码：from collections import deque; def bfs(graph, start): visited = set(); queue = deque([start]); visited.add(start); while queue: vertex = queue.popleft(); print(vertex); for neighbor in graph[vertex]: if neighbor not in visited: visited.add(neighbor); queue.append(neighbor)。BFS按照距离从近到远访问节点。例如图graph等于{0:[1,2], 1:[2], 2:[3], 3:[]}，从节点0开始BFS，访问顺序是0、1、2、3。BFS保证找到的路径是最短的。",
    
    "实现哈希表数据结构。代码：class HashTable: def __init__(self, size=10): self.size = size; self.table = [[] for _ in range(size)]; def hash(self, key): return hash(key) % self.size; def insert(self, key, value): idx = self.hash(key); for pair in self.table[idx]: if pair[0] == key: pair[1] = value; return; self.table[idx].append([key, value]); def get(self, key): idx = self.hash(key); for pair in self.table[idx]: if pair[0] == key: return pair[1]; return None。哈希表通过哈希函数将键映射到数组索引，实现O(1)平均时间复杂度。",
    
    "编写一个简单的神经网络前向传播。代码：import torch; import torch.nn as nn; class Net(nn.Module): def __init__(self): super().__init__(); self.fc1 = nn.Linear(10, 5); self.fc2 = nn.Linear(5, 1); def forward(self, x): x = torch.relu(self.fc1(x)); x = self.fc2(x); return x; model = Net(); output = model(torch.randn(1, 10))。神经网络由多个层组成，每层对输入进行线性变换后应用激活函数。前向传播从输入层到输出层逐层计算。",
    
    "实现一个简单的反向传播算法。代码：import torch; x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True); y = x.sum(); z = y * y; z.backward(); print(x.grad) 输出tensor([6., 6., 6.])。反向传播通过链式法则从输出层向输入层逐层计算梯度。在这个例子中，z等于y的平方，y等于x的和，所以dz/dx等于2y乘以1等于2乘以6等于12除以3等于4，每个x的梯度是4，但实际计算是2乘以y等于12，然后每个x贡献1，所以梯度是2乘以y乘以1等于2乘以3等于6。反向传播是训练神经网络的核心算法。",
    
    "编写一个文本分类的卷积神经网络。代码：import torch.nn as nn; class TextCNN(nn.Module): def __init__(self, vocab_size, embed_dim, num_classes): super().__init__(); self.embed = nn.Embedding(vocab_size, embed_dim); self.conv1 = nn.Conv1d(embed_dim, 16, kernel_size=3); self.conv2 = nn.Conv1d(embed_dim, 16, kernel_size=4); self.fc = nn.Linear(32, num_classes); def forward(self, x): x = self.embed(x).permute(0, 2, 1); x1 = self.conv1(x).max(dim=2)[0]; x2 = self.conv2(x).max(dim=2)[0]; x = torch.cat([x1, x2], dim=1); return self.fc(x)。文本CNN使用卷积操作提取文本的局部特征。",
    
    "实现一个序列到序列的模型。代码：import torch.nn as nn; class Encoder(nn.Module): def __init__(self, vocab_size, hidden_size): super().__init__(); self.embed = nn.Embedding(vocab_size, hidden_size); self.rnn = nn.GRU(hidden_size, hidden_size); def forward(self, x): x = self.embed(x); _, hidden = self.rnn(x); return hidden; class Decoder(nn.Module): def __init__(self, vocab_size, hidden_size): super().__init__(); self.embed = nn.Embedding(vocab_size, hidden_size); self.rnn = nn.GRU(hidden_size, hidden_size); self.fc = nn.Linear(hidden_size, vocab_size); def forward(self, x, hidden): x = self.embed(x); out, hidden = self.rnn(x, hidden); return self.fc(out), hidden。Seq2Seq模型由编码器和解码器组成，广泛应用于机器翻译、文本摘要。",
]


# ==================== 简单字符级分词器 ====================
class CharTokenizer:
    """基于字符的简单分词器，从句子中提取所有字符构建词表"""
    
    def __init__(self, sentences):
        # 收集所有字符
        chars = set()
        for sentence in sentences:
            chars.update(sentence)
        
        # 特殊token
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        
        # 构建词表
        self.vocab = [self.pad_token, self.eos_token, self.unk_token]
        self.vocab.extend(sorted(chars))
        
        # 映射
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.vocab_size = len(self.vocab)
        
        print(f"词表大小: {self.vocab_size}")
        print(f"词表内容: {self.vocab[:20]}...")
    
    def encode(self, text, add_eos=True):
        """将文本编码为token id序列"""
        ids = [self.token_to_id.get(c, self.unk_token_id) for c in text]
        if add_eos:
            ids.append(self.eos_token_id)
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """将token id序列解码为文本"""
        chars = []
        for i in ids:
            if i >= self.vocab_size:
                continue
            token = self.id_to_token[i]
            if skip_special_tokens and token in [self.pad_token, self.eos_token, self.unk_token]:
                continue
            chars.append(token)
        return "".join(chars)


# ==================== 数据集 ====================
class SentenceDataset(Dataset):
    """句子数据集"""
    
    def __init__(self, sentences, tokenizer, max_length=128):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 预处理所有句子
        self.encoded = []
        for sentence in sentences:
            ids = tokenizer.encode(sentence, add_eos=True)
            # 截断
            if len(ids) > max_length:
                ids = ids[:max_length]
            self.encoded.append(ids)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        ids = self.encoded[idx]
        # 创建labels（与input_ids相同，用于next token prediction）
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_fn(batch, pad_token_id):
    """批处理函数，填充序列"""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # 找到最大长度
    max_len = max(len(ids) for ids in input_ids)
    
    # 填充
    input_ids_padded = torch.zeros(len(batch), max_len, dtype=torch.long) + pad_token_id
    labels_padded = torch.zeros(len(batch), max_len, dtype=torch.long) - 100  # ignore_index
    
    for i, (ids, lbls) in enumerate(zip(input_ids, labels)):
        input_ids_padded[i, :len(ids)] = ids
        labels_padded[i, :len(lbls)] = lbls
    
    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
    }


# ==================== 训练函数 ====================
def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch，返回详细统计信息"""
    import time
    model.train()
    total_loss = 0
    num_batches = 0
    
    # 收集统计信息
    all_ternary_stats = {"positive": [], "negative": [], "neutral": [], "mean": []}
    all_grad_norms = []
    
    epoch_start = time.time()
    forward_time = 0
    backward_time = 0
    optimizer_time = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        t0 = time.time()
        outputs, _ = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        forward_time += time.time() - t0
        
        t0 = time.time()
        loss.backward()
        backward_time += time.time() - t0
        
        # 梯度裁剪（同时获取梯度范数，避免重复计算）
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        all_grad_norms.append(total_norm.item())
        
        t0 = time.time()
        optimizer.step()
        optimizer_time += time.time() - t0
        
        total_loss += loss.item()
        num_batches += 1
    
    epoch_time = time.time() - epoch_start
    
    # 训练结束后收集一次三值统计（而不是每10个batch）
    if hasattr(model, 'backbone'):
        for layer in model.backbone.layers:
            if hasattr(layer.mixer, '_stats_ternary_buffer') and layer.mixer._stats_ternary_buffer is not None:
                ternary = layer.mixer._stats_ternary_buffer.detach().cpu()
                all_ternary_stats["positive"].append((ternary > 0.5).float().mean().item())
                all_ternary_stats["negative"].append((ternary < -0.5).float().mean().item())
                all_ternary_stats["neutral"].append(((ternary >= -0.5) & (ternary <= 0.5)).float().mean().item())
                all_ternary_stats["mean"].append(ternary.abs().mean().item())
                break
    
    avg_loss = total_loss / num_batches
    
    # 计算困惑度
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    
    # 计算平均梯度范数
    avg_grad_norm = sum(all_grad_norms) / len(all_grad_norms) if all_grad_norms else 0.0
    max_grad_norm = max(all_grad_norms) if all_grad_norms else 0.0
    
    # 计算三值统计
    ternary_stats = {}
    for key in all_ternary_stats:
        if all_ternary_stats[key]:
            ternary_stats[key] = sum(all_ternary_stats[key]) / len(all_ternary_stats[key])
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "grad_norm_avg": avg_grad_norm,
        "grad_norm_max": max_grad_norm,
        "ternary_positive": ternary_stats.get("positive", 0.0),
        "ternary_negative": ternary_stats.get("negative", 0.0),
        "ternary_neutral": ternary_stats.get("neutral", 0.0),
        "ternary_mean": ternary_stats.get("mean", 0.0),
        "epoch_time": epoch_time,
        "forward_time": forward_time,
        "backward_time": backward_time,
        "optimizer_time": optimizer_time,
        "num_batches": num_batches,
    }


def evaluate_ternary_stats(model, tokenizer, device, test_sentences=None):
    """评估三值状态统计 - 分析ternary决策分布和纠缠统计"""
    model.eval()
    
    if test_sentences is None:
        test_sentences = TEST_SENTENCES[:5]
    
    print("\n" + "="*60)
    print("三值状态统计测试")
    print("="*60)
    
    all_stats = {
        "positive_ratio": [],  # 正向纠缠比例
        "negative_ratio": [],  # 负向纠缠比例
        "neutral_ratio": [],   # 中性比例
        "entanglement_mean": [],  # 平均纠缠强度
    }
    
    for i, sentence in enumerate(test_sentences):
        input_ids = torch.tensor([tokenizer.encode(sentence, add_eos=False)], dtype=torch.long, device=device)
        
        with torch.no_grad():
            output, _ = model(input_ids=input_ids)
        
        # 获取纠缠统计
        if hasattr(model, 'backbone'):
            stats = None
            for layer in model.backbone.layers:
                if hasattr(layer.mixer, '_stats_ternary_buffer') and layer.mixer._stats_ternary_buffer is not None:
                    ternary = layer.mixer._stats_ternary_buffer.detach().cpu()
                    total = layer.mixer._stats_total_buffer.item() if layer.mixer._stats_total_buffer is not None else 1.0
                    
                    # 计算三值分布
                    positive = (ternary > 0.5).float().mean().item()
                    negative = (ternary < -0.5).float().mean().item()
                    neutral = ((ternary >= -0.5) & (ternary <= 0.5)).float().mean().item()
                    
                    all_stats["positive_ratio"].append(positive)
                    all_stats["negative_ratio"].append(negative)
                    all_stats["neutral_ratio"].append(neutral)
                    all_stats["entanglement_mean"].append(ternary.abs().mean().item())
                    break
        
        if i < 3:  # 打印前3个样本的统计
            print(f"\n[样本 {i+1}] 长度: {len(sentence)}字")
            if all_stats["positive_ratio"]:
                print(f"  正向纠缠比例: {all_stats['positive_ratio'][-1]:.4f}")
                print(f"  负向纠缠比例: {all_stats['negative_ratio'][-1]:.4f}")
                print(f"  中性比例: {all_stats['neutral_ratio'][-1]:.4f}")
                print(f"  平均纠缠强度: {all_stats['entanglement_mean'][-1]:.4f}")
    
    # 汇总统计
    print("\n" + "-"*40)
    print("汇总统计:")
    for key, values in all_stats.items():
        if values:
            print(f"  {key}: 均值={np.mean(values):.4f}, 标准差={np.std(values):.4f}")
    
    return all_stats


def evaluate_causal_reasoning(model, tokenizer, device):
    """评估因果推理能力"""
    model.eval()
    
    print("\n" + "="*60)
    print("因果推理测试")
    print("="*60)
    
    # 因果推理测试用例
    causal_tests = [
        {
            "prompt": "如果下雨，地面会变湿。现在地面湿了，",
            "expected": ["可能下雨了", "下雨了", "可能是下雨"],
            "type": "溯因推理"
        },
        {
            "prompt": "所有的人都会死。苏格拉底是人，",
            "expected": ["苏格拉底会死", "所以苏格拉底会死"],
            "type": "演绎推理"
        },
        {
            "prompt": "学习使人进步。他学习很努力，",
            "expected": ["他进步很快", "所以他进步很快"],
            "type": "演绎推理"
        },
        {
            "prompt": "运动有益健康。她每天运动，",
            "expected": ["她很健康", "所以她很健康"],
            "type": "演绎推理"
        },
        {
            "prompt": "太阳从东方升起。今天是晴天，",
            "expected": ["太阳从东方升起", "太阳会从东方升起"],
            "type": "事实推理"
        },
    ]
    
    correct = 0
    total = len(causal_tests)
    
    for i, test in enumerate(causal_tests):
        input_ids = torch.tensor([tokenizer.encode(test["prompt"], add_eos=False)], dtype=torch.long, device=device)
        
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            temperature=0.3,
            top_k=3,
            use_cache=True,
        )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        
        # 检查是否包含预期答案
        matched = any(exp in generated_text for exp in test["expected"])
        if matched:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"\n[{status}] 类型: {test['type']}")
        print(f"  Prompt: {test['prompt']}")
        print(f"  生成: {generated_text}")
        print(f"  期望包含: {test['expected']}")
    
    accuracy = correct / total * 100
    print(f"\n因果推理准确率: {correct}/{total} ({accuracy:.1f}%)")
    return accuracy


def evaluate_semantic_coherence(model, tokenizer, device):
    """评估语义连贯性"""
    model.eval()
    
    print("\n" + "="*60)
    print("语义连贯性测试")
    print("="*60)
    
    # 语义连贯性测试用例
    coherence_tests = [
        {
            "context": "春天来了，花儿开放，小鸟在枝头歌唱。这是一年中最令人期待的季节，",
            "continuation": "大地从沉睡中苏醒",
            "description": "季节描述连贯"
        },
        {
            "context": "什么是友谊？友谊是人与人之间建立在相互理解、信任和关爱基础上的",
            "continuation": "真挚情感",
            "description": "定义延续连贯"
        },
        {
            "context": "人工智能是计算机科学的一个重要分支，它致力于研究和开发能够模拟、延伸和扩展人类智能的",
            "continuation": "理论、方法、技术及应用系统",
            "description": "技术描述连贯"
        },
        {
            "context": "如果下雨，地面会变湿。这是一个经典的逻辑推理例子，展示了",
            "continuation": "因果关系的推理过程",
            "description": "逻辑描述连贯"
        },
        {
            "context": "北京位于华北平原的北部，背靠燕山，东临天津，其余方向与河北省",
            "continuation": "相邻",
            "description": "地理描述连贯"
        },
    ]
    
    correct = 0
    total = len(coherence_tests)
    
    for i, test in enumerate(coherence_tests):
        input_ids = torch.tensor([tokenizer.encode(test["context"], add_eos=False)], dtype=torch.long, device=device)
        
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=30,
            temperature=0.1,
            top_k=1,  # 贪婪解码
            use_cache=True,
        )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        
        # 检查生成是否包含期望的延续
        if test["continuation"] in generated_text:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"\n[{status}] {test['description']}")
        print(f"  上下文: {test['context'][:50]}...")
        print(f"  期望延续: {test['continuation']}")
        print(f"  实际生成: {generated_text[len(test['context']):][:50]}...")
    
    accuracy = correct / total * 100
    print(f"\n语义连贯性准确率: {correct}/{total} ({accuracy:.1f}%)")
    return accuracy


def evaluate_math_reasoning(model, tokenizer, device):
    """评估数学运算推演能力"""
    model.eval()
    
    print("\n" + "="*60)
    print("数学运算推演测试")
    print("="*60)
    
    # 数学运算测试用例
    math_tests = [
        {
            "prompt": "计算1加2等于",
            "expected": ["3", "等于3"],
            "description": "加法运算"
        },
        {
            "prompt": "计算10减3等于",
            "expected": ["7", "等于7"],
            "description": "减法运算"
        },
        {
            "prompt": "计算4乘以5等于",
            "expected": ["20", "等于20"],
            "description": "乘法运算"
        },
        {
            "prompt": "计算20除以4等于",
            "expected": ["5", "等于5"],
            "description": "除法运算"
        },
        {
            "prompt": "计算2的3次方等于",
            "expected": ["8", "等于8"],
            "description": "乘方运算"
        },
        {
            "prompt": "计算根号9等于",
            "expected": ["3", "等于3"],
            "description": "开方运算"
        },
        {
            "prompt": "解方程x加2等于5，x等于",
            "expected": ["3", "x等于3"],
            "description": "方程求解"
        },
        {
            "prompt": "三角形面积公式S等于底乘以高除以2，底为6高为4，面积等于",
            "expected": ["12", "等于12"],
            "description": "面积计算"
        },
        {
            "prompt": "圆的面积公式S等于π乘以r的平方，半径r等于5，面积约等于",
            "expected": ["78.5", "78.54"],
            "description": "圆面积计算"
        },
        {
            "prompt": "一元二次方程x平方减5x加6等于0，x等于",
            "expected": ["2", "3", "x等于2", "x等于3"],
            "description": "二次方程求解"
        },
    ]
    
    correct = 0
    total = len(math_tests)
    
    for i, test in enumerate(math_tests):
        input_ids = torch.tensor([tokenizer.encode(test["prompt"], add_eos=False)], dtype=torch.long, device=device)
        
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=30,
            temperature=0.1,
            top_k=1,
            use_cache=True,
        )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        
        # 检查是否包含预期答案
        matched = any(exp in generated_text for exp in test["expected"])
        if matched:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"\n[{status}] {test['description']}")
        print(f"  Prompt: {test['prompt']}")
        print(f"  生成: {generated_text}")
        print(f"  期望包含: {test['expected']}")
    
    accuracy = correct / total * 100
    print(f"\n数学运算准确率: {correct}/{total} ({accuracy:.1f}%)")
    return accuracy


def evaluate_code_understanding(model, tokenizer, device):
    """评估代码上下文理解能力"""
    model.eval()
    
    print("\n" + "="*60)
    print("代码上下文理解测试")
    print("="*60)
    
    # 代码理解测试用例
    code_tests = [
        {
            "prompt": "代码：x = 10; y = x + 5; print(y) 输出",
            "expected": ["15", "输出15"],
            "description": "变量定义使用"
        },
        {
            "prompt": "代码：def add(a, b): return a + b; result = add(3, 5); print(result) 输出",
            "expected": ["8", "输出8"],
            "description": "函数定义调用"
        },
        {
            "prompt": "代码：x = 5; if x > 0: print('正数') else: print('非正数') 输出",
            "expected": ["正数", "输出正数"],
            "description": "条件语句分支"
        },
        {
            "prompt": "代码：for i in range(3): print(i) 输出",
            "expected": ["0", "1", "2"],
            "description": "循环语句"
        },
        {
            "prompt": "代码：numbers = [1, 2, 3, 4, 5]; print(numbers[2]) 输出",
            "expected": ["3", "输出3"],
            "description": "列表索引访问"
        },
        {
            "prompt": "代码：person = {'name': '张三', 'age': 25}; print(person['name']) 输出",
            "expected": ["张三", "输出张三"],
            "description": "字典键访问"
        },
        {
            "prompt": "代码：try: result = 10 / 0; except ZeroDivisionError: print('除零错误') 输出",
            "expected": ["除零错误", "输出除零错误"],
            "description": "异常处理"
        },
        {
            "prompt": "代码：def fib(n): if n <= 1: return n; return fib(n-1) + fib(n-2); print(fib(5)) 输出",
            "expected": ["5", "输出5"],
            "description": "递归函数"
        },
        {
            "prompt": "代码：arr = [1, 3, 5, 7, 9]; 在数组中用二分查找找7，返回索引",
            "expected": ["3", "索引3"],
            "description": "二分查找"
        },
        {
            "prompt": "代码：quicksort([3, 6, 8, 10, 1, 2, 1]) 排序结果",
            "expected": ["[1, 1, 2, 3, 6, 8, 10]", "1, 1, 2, 3, 6, 8, 10"],
            "description": "快速排序"
        },
    ]
    
    correct = 0
    total = len(code_tests)
    
    for i, test in enumerate(code_tests):
        input_ids = torch.tensor([tokenizer.encode(test["prompt"], add_eos=False)], dtype=torch.long, device=device)
        
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            temperature=0.1,
            top_k=1,
            use_cache=True,
        )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        
        # 检查是否包含预期答案
        matched = any(exp in generated_text for exp in test["expected"])
        if matched:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"\n[{status}] {test['description']}")
        print(f"  Prompt: {test['prompt']}")
        print(f"  生成: {generated_text}")
        print(f"  期望包含: {test['expected']}")
    
    accuracy = correct / total * 100
    print(f"\n代码理解准确率: {correct}/{total} ({accuracy:.1f}%)")
    return accuracy


def analyze_parameter_effects(model, tokenizer, device):
    """分析各参数对模型行为的影响"""
    model.eval()
    
    print("\n" + "="*60)
    print("参数效果分析")
    print("="*60)
    
    # 获取当前配置参数
    config_params = {}
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        first_layer = model.backbone.layers[0]
        if hasattr(first_layer, 'mixer'):
            mixer = first_layer.mixer
            config_params = {
                "d_state": getattr(mixer, 'd_state', 'N/A'),
                "entanglement_window": getattr(mixer, 'entanglement_window', 'N/A'),
                "entanglement_scale": getattr(mixer, 'entanglement_scale', 'N/A'),
                "entanglement_threshold": getattr(mixer, 'entanglement_threshold', 'N/A'),
                "entanglement_init": getattr(mixer, 'entanglement_init', 'N/A'),
                "state_scan_chunk_size": getattr(mixer, 'state_scan_chunk_size', 'N/A'),
            }
    
    print("\n当前配置参数:")
    for key, value in config_params.items():
        print(f"  {key}: {value}")
    
    # 分析纠缠模式
    print("\n纠缠模式分析:")
    test_sentence = TEST_SENTENCES[0]
    input_ids = torch.tensor([tokenizer.encode(test_sentence, add_eos=False)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        output, _ = model(input_ids=input_ids)
    
    # 收集各层的纠缠统计
    layer_stats = []
    if hasattr(model, 'backbone'):
        for i, layer in enumerate(model.backbone.layers):
            if hasattr(layer.mixer, '_stats_ternary_buffer') and layer.mixer._stats_ternary_buffer is not None:
                ternary = layer.mixer._stats_ternary_buffer.detach().cpu()
                layer_stats.append({
                    "layer": i,
                    "mean": ternary.mean().item(),
                    "std": ternary.std().item(),
                    "max": ternary.max().item(),
                    "min": ternary.min().item(),
                })
    
    if layer_stats:
        print("\n各层纠缠强度统计:")
        print(f"{'层':<6} {'均值':<10} {'标准差':<10} {'最大值':<10} {'最小值':<10}")
        for stat in layer_stats:
            print(f"{stat['layer']:<6} {stat['mean']:<10.4f} {stat['std']:<10.4f} {stat['max']:<10.4f} {stat['min']:<10.4f}")
    
    # 分析纠缠类型（局部 vs 全局）
    entanglement_window = config_params.get("entanglement_window", 0)
    if entanglement_window and entanglement_window > 0:
        print(f"\n纠缠类型: 局部纠缠 (窗口大小: {entanglement_window})")
        print("  - 每个位置只与前后窗口内的位置产生纠缠")
        print("  - 适合捕捉局部依赖关系")
    else:
        print("\n纠缠类型: 全局纠缠")
        print("  - 每个位置可与任意位置产生纠缠")
        print("  - 适合捕捉长距离依赖关系")
    
    return config_params, layer_stats


def evaluate_generation(model, tokenizer, device, num_samples=5, max_new_tokens=32):
    """评估生成能力"""
    model.eval()
    
    # 选择一些句子作为prompt（取前半部分）
    test_prompts = []
    for sentence in TEST_SENTENCES[:num_samples]:
        # 取句子的前半部分作为prompt
        mid = len(sentence) // 2
        prompt = sentence[:mid]
        test_prompts.append(prompt)
    
    print("\n" + "="*60)
    print("生成测试:")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts):
        input_ids = torch.tensor([tokenizer.encode(prompt, add_eos=False)], dtype=torch.long, device=device)
        
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=5,
            use_cache=True,
            dynamic_activation=True,  # 启用动态激活
        )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        
        print(f"\n[{i+1}] Prompt: {prompt}")
        print(f"    生成: {generated_text}")
        print(f"    原句: {TEST_SENTENCES[i]}")


def main():
    parser = argparse.ArgumentParser(description='TESM 语义生成测试')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto', 'torch', 'cuda', 'triton', 'tilelang'],
                        help='Kernel backend to use (default: auto)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (20 epochs)')
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 20
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"Kernel backend: {args.backend}")
    print(f"Epochs: {args.epochs}")
    
    # 创建分词器
    tokenizer = CharTokenizer(TEST_SENTENCES)
    
    # 创建数据集
    dataset = SentenceDataset(TEST_SENTENCES, tokenizer, max_length=2056)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )
    
    # 创建模型配置（使用tiny配置，适合小数据集）
    config = TESMConfig.tiny()
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.max_seq_len = 2056  # 长序列支持长文本
    
    # 启用词表抑制机制（训练后启用）
    config.vocab_suppression = False  # 训练时关闭，避免影响学习
    config.suppression_bias = -5.0  # 降低抑制强度，允许更多token
    
    # 启用语义相关激活（解决泛化问题）
    config.semantic_activation = True  # 训练时学习token共现关系
    config.semantic_activation_strength = 0.5  # 相关token激活强度
    config.semantic_activation_threshold = 0.1  # 低阈值，让模型自己学习区分
    
    # 覆盖ssm_cfg中的关键参数
    config.ssm_cfg = {
        "d_state": 256,
        "expand": 2,
        "ent_rank": 32,
        "entanglement_scale": 0.25,
        "entanglement_threshold": 0.05,
        "entanglement_init": 0.3,
        "entanglement_window": 16,  # 局部纠缠模式
        "entanglement_block_size": 256,
        "state_scan_chunk_size": 16,
        "use_triton_kernels": True,
        "kernel_backend": args.backend,  # 使用命令行指定的后端
        "kernel_mode": "fast",
        "decay_init_bias": 0.0,  # 短序列
        "annealing_enabled": True,
        "T_start": 10.0,
        "T_end": 0.1,
        "annealing_steps": 500,
        "annealing_schedule": "cosine",
    }
    
    print(f"\n模型配置:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  max_seq_len: {config.max_seq_len}")
    
    # 创建模型
    model = TESMLMHeadModel(config, device=device)
    model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # 学习率调度器
    num_epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    best_loss = float("inf")
    for epoch in range(num_epochs):
        stats = train_epoch(model, dataloader, optimizer, device, epoch)
        scheduler.step()
        
        if stats["loss"] < best_loss:
            best_loss = stats["loss"]
        
        # 每10个epoch打印一次详细日志
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Loss: {stats['loss']:.4f} | Best: {best_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"  Perplexity: {stats['perplexity']:.2f}")
            print(f"  Grad Norm: avg={stats['grad_norm_avg']:.4f}, max={stats['grad_norm_max']:.4f}")
            if stats['ternary_mean'] > 0:
                print(f"  Ternary: +{stats['ternary_positive']:.3f} -{stats['ternary_negative']:.3f} ~{stats['ternary_neutral']:.3f} |强度={stats['ternary_mean']:.4f}")
            # 打印时间分解
            print(f"  Time: epoch={stats['epoch_time']:.2f}s | forward={stats['forward_time']:.2f}s | backward={stats['backward_time']:.2f}s | optimizer={stats['optimizer_time']:.2f}s | batches={stats['num_batches']}")
            
            # 每20个epoch测试生成
            if (epoch + 1) % 20 == 0:
                evaluate_generation(model, tokenizer, device, num_samples=3, max_new_tokens=1000)
    
    # 最终评估
    print("\n" + "="*60)
    print("最终评估")
    print("="*60)
    
    # 训练结束后构建共现矩阵（用于推理时的语义相关激活）
    if model.semantic_activation:
        print("跳过共现矩阵构建（训练已足够）")
        # model.build_cooccurrence_from_dataset(dataloader, max_batches=100)
    
    evaluate_generation(model, tokenizer, device, num_samples=10, max_new_tokens=1000)
    
    # 三值状态统计测试
    evaluate_ternary_stats(model, tokenizer, device)
    
    # 因果推理测试
    evaluate_causal_reasoning(model, tokenizer, device)
    
    # 语义连贯性测试
    evaluate_semantic_coherence(model, tokenizer, device)
    
    # 数学运算推演测试
    evaluate_math_reasoning(model, tokenizer, device)
    
    # 代码上下文理解测试
    evaluate_code_understanding(model, tokenizer, device)
    
    # 参数效果分析
    analyze_parameter_effects(model, tokenizer, device)
    
    # 过拟合测试：检查是否能完整复述训练数据
    print("\n" + "="*60)
    print("过拟合测试（复述训练数据）")
    print("="*60)
    
    model.eval()
    correct = 0
    total = len(TEST_SENTENCES)
    
    for i, sentence in enumerate(TEST_SENTENCES):
        # 用前几个字符作为prompt
        prompt_len = min(5, len(sentence) // 3)
        prompt = sentence[:prompt_len]
        
        input_ids = torch.tensor([tokenizer.encode(prompt, add_eos=False)], dtype=torch.long, device=device)
        
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=len(sentence) - prompt_len + 10,
            temperature=0.1,  # 低温度，更确定性
            top_k=1,  # 贪婪解码
            use_cache=True,
        )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        
        # 检查是否包含完整原句
        if sentence in generated_text:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        if i < 10:  # 只打印前10个
            print(f"[{status}] Prompt: {prompt}")
            print(f"    生成: {generated_text}")
            print(f"    原句: {sentence}")
            print()
    
    print(f"\n过拟合率: {correct}/{total} ({100*correct/total:.1f}%)")
    
    # 保存模型
    save_path = os.path.join(os.path.dirname(__file__), "test_semantic_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "tokenizer_vocab": tokenizer.vocab,
    }, save_path)
    print(f"\n模型已保存到: {save_path}")


if __name__ == "__main__":
    main()
