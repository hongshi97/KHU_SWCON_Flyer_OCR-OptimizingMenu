import os
import string
import argparse
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from OCR_modules.utils import CTCLabelConverter, AttnLabelConverter
from OCR_modules.dataset import RawDataset, AlignCollate
from OCR_modules.model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### option 정의
parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', default='test_images/', help='path to image_folder which contains text images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
parser.add_argument('--saved_model', default='/content/drive/MyDrive/데이터분석캡스톤디자인/CLOVAOCR4/saved_models/TPS-ResNet-BiLSTM-CTC-Seed3333/best_accuracy.pth', help="path to saved_model to evaluation")
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default=' !"#%&\'()*+,-./0123456789:<=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz~가각간갇갈갉감갑값갓갔강갖같갚갛개객갠갤갬갭갯갱갸갼걀걍거걱건걷걸검겁것겅겉게겐겔겜겟겠겡겨격겪견결겸겹겼경곁계고곡곤곧골곰곱곳공곶과곽관괄괌광괘괜괭괴괸굄굉교굣구국군굳굴굶굼굽굿궁궂궈권궐궤귀귄귐귓규균귤그극근글금급긋긍기긱긴긷길김깁깃깅깊까깍깎깐깔깜깝깡깥깨깬깻깽꺄꺼꺽꺾껀껄껌껍껏껑께껜껨껴껸꼈꼐꼬꼭꼰꼴꼼꼽꽁꽂꽃꽈꽉꽐꽤꽥꽹꾀꾸꾹꾼꿀꿇꿈꿉꿍꿔꿨꿩꿰뀀뀌뀐뀔뀝뀨끄끈끊끌끎끓끔끗끙끝끼끽낀낄낌낑나낙낚난날낡남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냘냠냥너넉넋넌널넓넘넙넛넜넝넣네넥넨넬넴넵넷넹녀녁년념녔녕녘녜노녹논놀놈놉놋농높놓놔뇌뇨뇽누눅눈눌눔눕눗눙눠뉘뉜뉴늄느늑는늘늙늠능늦늪늬니닉닌닐님닙닛닝닢다닥닦단닫달닭닮담답닷당닻닿대댁댄댈댐댑댓댔댕댜더덕던덜덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎅뎌뎐도독돈돋돌돔돕돗동돛돝돼됐되된될됨됩두둑둔둘둠둡둣둥둬뒀뒤뒷뒹듀듄듈드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딩딪따딱딴딸땀땁땅땋때땍땐땔땜땠땡떠떡떤떨떳떴떵떻떼떽뗌뗏뗐뗑또똑똔똘똥뙈뚜뚝뚠뚤뚫뚱뛰뛴뛸뜀뜨뜩뜬뜯뜰뜸뜻띄띠띤띨띵라락란랄람랍랏랐랑랗래랙랜랠램랩랫랬랭랴략랸량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례로록론롤롬롭롯롱롸뢰료룔룡루룩룬룰룸룹룻룽뤄뤠뤼륀륄류륙륜률륨륫륭르륵른를름릅릇릉릎리릭린릴림립릿링마막만많맏말맑맘맙맛망맞맡맣매맥맨맴맵맷맹맺먀머먹먼멀멈멋멍메멕멘멜멤멥멧멩며멱면멸명몇모목몫몬몰몸몹못몽뫼묘무묵묶문묻물묾뭄뭇뭉뭍뭐뭔뭘뭣뮈뮌뮐뮤뮨뮬므믄믈믐미믹민믿밀밈밋밌밍및밑바박밖반받발밝밟밤밥밧방밭배백밴밸뱀뱁뱃뱅뱉뱍뱐버벅번벌범법벗벙벚베벡벤벧벨벰벱벳벵벼벽변별볍볏병볕보복볶본볼봄봅봇봉봐봤봬뵈뵙부북분불붉붐붑붓붕붙붚뷔뷜뷰브븍븐블븜븟비빅빈빌빔빕빗빙빚빛빠빡빤빨빰빱빳빵빻빼빽뺀뺄뺍뺑뺨뻐뻑뻔뻗뻘뻣뻤뻥뻬뼈뼘뼛뼝뽀뽁뽄뽈뽐뽑뽕뾰뿅뿌뿍뿐뿔뿜뿡쁘쁜쁠쁨삐삑삘삠사삭산살삶삼삽삿샀상새색샌샐샘샙샛생샤샥샨샬샴샵샷샹섀서석섞선설섧섬섭섯섰성섶세섹센셀셈셉셋셍셔션셜셧셨셩셰셴셸소속손솔솜솝솟송솥솨쇄쇠쇳쇼숀숄숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉐쉑쉔쉘쉬쉭쉰쉴쉼쉽슁슈슉슐슘슝스슥슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌤쌩썅써썩썬썰썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏨쏭쏴쐐쑈쑤쑥쑨쓰쓱쓴쓸씀씁씌씨씩씬씰씸씹씻씽아악안앉않알앎앓암압앗았앙앞애액앤앨앰앱앳앵야약얀얄얇얌얍얏양얕얗얘얜어억언얹얻얼얽엄업없엇었엉엌엎에엑엔엘엠엡엣엥여역엮연열염엽엿였영옅옆예옌옐옙옛옜오옥온올옮옳옴옵옷옹옻와왁완왈왑왓왔왕왜왠왱외왼요욕욘욜욤욥욧용우욱운울움웁웃웅워웍원월웜웠웨웩웬웰웸웹위윅윈윌윔윗윙유육윤율윰융윷으윽은을읊음읍응읔읖의이익인일읽잃임입잇있잉잊잎자작잔잖잗잘잠잡잣잤장잦재잭잰잼잽잿쟁쟈쟉쟌쟝저적전절젊점접젓정젖제젝젠젤젬젯져젼졌졔조족존졸좀좁종좇좋좌죄죠죤주죽준줄줌줍줏중줘줬쥐쥑쥔쥘쥬쥰쥴즈즉즌즐즘즙증지직진짇질짊짐집짓징짖짙짚짜짝짠짤짧짬짭짱째짼쨈쨋쨌쨍쩌쩍쩐쩔쩜쩝쩡쩨쪄쪘쪼쪽쫀쫄쫌쫑쫓쫘쭈쭉쭌쭘쭝쮸쯔쯤찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻창찾채책챈챌챔챕챗챙챠챤챨처척천철첨첩첫청체첸첼쳄쳅쳇쳐쳔쳤쳬초촉촌촘촛총촨촬최쵸추축춘출춤춥춧충춰췄췌취췬츄츈츠측츨츰층치칙친칠칡침칩칫칭카칵칸칼캄캅캇캉캐캔캘캠캡캣캥캬커컨컬컴컵컷컸컹케켄켈켐켑켓켜켠켤켰코콕콘콜콤콥콧콩콰콴콸쾅쾌쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀘퀴퀵퀸퀼큅큐큘크큰클큼킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탕태택탠탤탬탭탯탱터턱턴털텀텁텃텅테텍텐텔템텝텟텨텬톈토톡톤톨톰톱톳통톺퇘퇴툇투툭툰툴툼툽퉁튀튄튈튑튜튠튤튬트특튼틀틈틉틋틔티틱틴틸팀팁팃팅파팍팎판팔팜팝팟팠팡팥패팩팬팰팸팹팻팽퍼펀펄펌펍펏펑페펙펜펠펨펫펭펴편펼평폐포폭폰폴폼폿퐁푀표푸푹푼풀풂품풉풋풍퓌퓨퓰퓸프픈플픔피픽핀필핌핍핏핑하학한할함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혔형혜호혹혼홀홈홉홋홍화확환활홧황횃회획횟횡효후훅훈훌훔훗훙훠훤훨훼휀휄휘휙휜휠휩휴휼흉흐흑흔흘흙흠흡흥흩희흰히힉힌힐힘힙힛힝', help='character label')
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
options = parser.parse_args(args=[])


# output 함수정의
def outputs(opt):
    output,img_names = [], []
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    
    
    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    
    
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    
    
    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                img_names.append(img_name)
                output.append(pred)

    df = pd.DataFrame({'image_names':img_names, 'outputs':output})

    return(df)


# 정리
def Prediction(option):

    if option.sensitive:
        option.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    option.num_gpu = torch.cuda.device_count()

    return outputs(option)