# UniOCR

---
## 1. CRAFT : Detection 모델

CRAFT는 Character Region Awareness for Text Detection의 약자로 Naver의 Clova AI 팀에서 발표한 OCR을 위한 문자 단위 Text Detection에 관한 논문에서 제시한 deep learning 모델이다.

CRAFT 알고리즘은 Scene Text Detection(SDT) 문제를 해결하기 위해 기본적으로 딥러닝 기반 모델을 이용한다.

![](https://miro.medium.com/max/700/1*z4EDHfQrYwltnCdXoo3eyQ.jpeg)

모델의 목적은 입력으로 사용한 이미지에 대해 각 픽셀마다 Region Score와 Affinity Score를 예측하는 것이다. Region Score는 해당 픽셀이 문자의 중심일 확률을 의미한다. Affinity Score는 해당 픽셀의 인접한 두 문자의 중심일 확률을 의미한다. 이 점수를 기반으로 개별 문자가 하나의 단어로 그룹화될 것인지 결정한다. 

![](https://miro.medium.com/max/700/1*dh1tP51LL3QlPJvV7Anc-Q.jpeg)

이미지처럼 Region Score는 각 픽셀이 문자의 중심에 가까울수록 확률이 1에 가깝고 문자 중심에서 멀어질 수록 확률이 0에 가깝도록 모델을 학습하는 것이다. 이와 동시에 동일 모델은 각 픽셀이 인접한 문자의 중심에 가까울 확률인 Affinity Score도 예측할 수 있어야 한다.
잘 학습된 모델을 통해 입력 이미지의 Region score와 Affinity score를 예측하면 최종적으로 다음과 같은 Text Detection 결과를 얻을 수 있다.

![](https://miro.medium.com/max/700/1*38BUBJ7F2yGim7WDevA-BA.jpeg)


### CRAFT 모델의 Architecture

![](https://miro.medium.com/max/2000/1*Bk8B872CriHl_DzrsW4SRQ.png)
왼쪽은 논문에 제시되어 있는 CRAFT 네트워크 구조이고, 오른쪽은 github에 공개되어 있는 코드를 통해 재구성한 CRAFT 네트워크 구조다.

CRAFT 네트워크 구조는 다음과 같은 특징을 갖는다.

* Fully Convolutional network architecture
* VGG-16을 기반으로 함
* Batch normalization을 사용함
* U-net 처럼 UpConv를 통해 Low-level 특징을 집계하여 예측함

VGG-16은 총 16개 레이어로 구성되어 있으며 3x3 필터를 사용한다.(1x1 필터가 포함되기도 함) 활성화 함수로는 ReLU를 사용하며, Stride와 Padding은 각각 1로 설정한다.
CRAFT 모델의 출력은 Classification이 아닌 Image 형태이기 때문에 기존의 VGG-16을 Fully ConvNet으로 변형해 사용한다.

![](https://miro.medium.com/max/700/1*F2NobIwznjcOPcRdGxl9ZA.png)

사실 CRAFT의 핵심적인 부분은 Hidden Layer의 구성보다는 output layer 및 학습 방법에 있다.

craft.py
```python
class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
	    super(CRAFT, self).__init__()
		
		""" Base network """
		self.basenet = vgg16_bn(pretrained, freeze)
		
```

논문에 명시된 네트워크 구조와 같이 CRAFT 코드에서도 VGG16-BN을 기본틀로 사용한다.

![](https://miro.medium.com/max/700/1*w_46sFeAAPnq5JP3uX8wHQ.png)

pytorch에서는 VGG를 기본 모듈로 포함하고 있으며 위의 구성에서 A, B, D, E에 해당하는 네트워크 및 각각에 Batch-Norm이 포함된 버전의 네트워크를 선택할 수 있다.

* A: 11-layers [Batch-Norm]
* B: 13-layers [Batch-Norm]
* D: 16-layers [Batch-Norm]
* E: 19-layers [Batch-Norm]

pytorch에서 제공하는 VGG16-BN의 구성은 다음과 같이 도식화할 수 있다.

![](https://miro.medium.com/max/2000/1*Tv8Twm3cQWSEMbbUySN3KQ.png)
VGG는 크게 Feature Extractor 부분과 Classifier 부분으로 나눌 수 있다.

![](https://miro.medium.com/max/1400/1*O2hKJwoSqBgaIs0np11qpQ.png)
VGG16 with Batch-Normalization

CRAFT는 VGG16-BN의 Feature Extractor 부분을 변형하여 사용한다.

```python
class vgg16_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True):
	    super(vgg16_bn, self).__init__()
		model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
		vgg_pretrained_features = models.vgg16_bn(pretrained+pretrained).features
		self.slice1 = torch.nn.Sequential()
		self.slice2 = torch.nn.Sequential()
		self.slice3 = torch.nn.Sequential()
		self.slice4 = torch.nn.Sequential()
		self.slice5 = torch.nn.Sequential()
		for x in range(12):         # conv2_2
		    self.slice1.add_module(str(x), vgg_pretrained_features[x])
		for x in range(12, 19):         # conv3_3
		    self.slice2.add_module(str(x), vgg_pretrained_features[x])
		for x in range(19, 29):         # conv4_3
		    self.slice3.add_module(str(x), vgg_pretrained_features[x])
		for x in range(29, 39):         # conv5_3
		    self.slice4.add_module(str(x), vgg_pretrained_features[x])
```

여기서 [0:12], [12:19], [19:29], [29,39]는 VGG16-BN의 Feature Extractor를 구성하는 모듈들의 부분집합으로 CNN, Batch-Norm, ReLU, MaxPooling들이 포함된다.

![](https://miro.medium.com/max/700/1*rhU5kQD_UsWC07KawxA0fQ.png)
VGG16-BN의 Feature Extractor

![](https://miro.medium.com/max/700/1*yBf-lHYgwj9jB1WLV24DUQ.png)
VGG16-BN의 Feature Extractor는 총 43개 모듈로 구성되어 있다.

CRAFT는 VGG16-BN의 Feature Extractor 구성 중 앞의 39(0 ~ 38)개 모듈 뒤에 3개 모듈(MaxPooling-Conv-Conv)를 붙인 변형을 뼈대로 사용한다.

```python
        # fc6, fc7 without atrous conv
		self.slice5 = torch.nn.Sequential(
		    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
			nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
			nn.Conv2d(1024, 1024, kernel_size=1)
		)
```
slice5 부분에서 입력과 출력의 크기 변화가 없다


![](https://miro.medium.com/max/700/1*uC8YngYsI8gd-ipoGkV4sA.png)
CRAFT의 Backbone

마지막에 이어붙이 MaxPooling-Dilated Conv-1x1Conv는 입력의 크기를 줄이지 않으면서 채널 수만 늘린다. 또한 비선형 함수(ReLU)를 사용하지 않는다.

![](https://miro.medium.com/max/688/1*ZDbg-VpMp2NP2ZozzntE5A.png)
https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5

이것으로 앞에서 제시한 CRAFT basenet이 결정된다.

```python
class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
	    super(CRAFT, self).__init__()
		
		""" Base network """
		self.basenet = vgg16_bn(pretrained, freeze)
```

basenet에는 5개의 slice가 있다는 것을 기억하자, 각 부분의 출력은 Feature Extractor의 깊이 별 특징 맵에 해당한다. 
해상도 및 추상화 정도가 서로 다른 특징 맵들 U-Net과 같이 결합하기 위해 CRAFT의 forward 부분은 다음과 같이 구현되어 있다.

![](https://miro.medium.com/max/2000/1*qU4NPWxfhzY-LNdmL4qfjg.png)

이때, UpConv는 Conv-BatchNorm-ReLu-Conv-BatchNorm-ReLU로 구성된다.

![](https://miro.medium.com/max/2120/1*pUEdYb89fnNTe6vJ5mwAtw.png)
![](https://miro.medium.com/max/956/1*BHc7LIM8jflB3v0_wbkzSA.png)

주목할 부분은 Up-conv의 입력채널 수와 출력 채널 수이다.
첫번째 Conv2d의 입력 채널 수는 파라미터로 전달받은 in_ch와 mod_ch의 합이다. 출력 채널은 out_ch이다. 따라서, UpConv는 (in_ch + mid_ch) 채널 수를 out_ch 채널 수로 바꾼다.

![](https://miro.medium.com/max/1400/1*z1iAxktkWbdcLBedDVS84w.png)

즉, upconv1를 거치면 특징맵의 (1024+512)채널은 256채널로 줄어든다.
실제 구현에서 up-conv 부분의 채널 수를 보면, 논문에 명시된 네트워크 구조 그림과는 차이가 있다.

![](https://miro.medium.com/max/659/1*-kFdZNerp5k89E50br1new.png)
DownConv 마지막 출력 채널수는 1024

마지막으로 Feature Extractor의 특징 맵으로부터 Region score와 Affinity score의 dense map을 추출하기 위한 Output Layer를 살펴보자.

![](https://miro.medium.com/max/1874/1*y5gl6fzESHL597BpyuR9Gw.png)
![](https://miro.medium.com/max/3010/1*5gHYAwofIwMwxWTyxf4sBQ.png)

그림에서는 마지막 출력층의 conv[1x1x2] 레이어가 갈라지는 화살표로만 표시되어 있다.
공개된 소스코드를 기반으로 논문의 네트워크 구조 그림을 다음과 같이 재구성할 수 있다.

![](https://miro.medium.com/max/1400/1*b6I-Bdj5itX7tllJ5HRKbg.png)



### 정답 레이블 제작은?

![](https://miro.medium.com/max/700/1*ollTTgqlKd2NvSVrsvBecQ.png)

CRAFT 모델은 위 이미지처럼 입력한 이미지에 대해 Region score map과 Affinity score map을 출력한다. 따라서, 학습에는 각 입력으로 사용될 이미지들의 기대 출력에 해당하는 기준 정답(ground-truth)이 필요하다. 특정 이미지에 대한 ground-truth를 생성하는 과정은 다음과 같다.

![](https://miro.medium.com/max/700/1*UuuULnVIGpIoe8p63zLXbA.png)


#### Region Score Map 생성하기
1. 2차원 isotropic gaussian map을 준비한다.
2. 이미지 내의 각 문자에 대해 경계상자를 그리고 해당 경계상자에 맞춰 가우시안 맵을 변형시킨다.
3. 변형된 가우시안 맵을 원본 이미지의 경계상자 좌표와 대응되는 위치에 해당하는 Label Map 좌표에 할당한다.


![](https://miro.medium.com/max/644/1*CsAIYlZgSAW8Ikmcez-BaA.png)
2D isotropic gaussian map 준비

![](https://miro.medium.com/max/700/1*BEes7mM33y1bNkZbXSCILw.png)
원본 이미지의 절반 사이즈에 해당하는 region score map을 만들고 모든 픽셀을 0으로 초기화

![](https://miro.medium.com/max/700/1*3NPKEBg0XxIdzsAWgg3RbA.png)
원본 이미지의 문자에 대한 경계상자를 그렸을 때, 대응되는 region score map 좌표에 경계상자에 맞춰 변형시킨 가우시안 맵을 할당한다.

![](https://miro.medium.com/max/700/1*ziiFqxddovXWbwbzc1fuYQ.png)
각각의 문자에 대해 동일한 과정을 거쳐 오른쪽과 같은 region score에 대한 ground-truth를 만든다.


#### Affinity score map 생성하기
Affinity score map은 region score map을 생성할 때 사용했던 경계상자를 활용한다.

앞에서 언급한 '인접한 두 문자'의 정의를 명확히 해보자.

![](https://miro.medium.com/max/700/1*mMIx7TKvo6eo1MZLKmA9KQ.png)
문자(character) 경계상자에 대각선을 그엇을 때 생기는 위쪽 삼각형과 아래쪽 삼각형으로부터 각 중심점을 구할 수 있다. 하나의 단어에 포함되는 인접한 두 문자에 대해 삼각형 중심점을 이었을 때 만들어지는 사각형이 바로 affinity box가 된다.

![](https://miro.medium.com/max/700/1*K_PYcPCchhAtPBkTut3MIA.png)
region score map 생성시와 마찬가지로 원본 이미지의 affinity box 좌표에 대응되는 affinity score map의 좌표에 경계상자에 맞춰 변형시킨 가우시안 맵을 할당한다.

![](https://miro.medium.com/max/700/1*MX-ty863xz-AlvnmP7DB9g.png)
각각의 문자에 대해 동일한 과정을 거쳐 오른쪽과 같은 affinity score에 대한 ground-truth를 만들 수 있다.

결국 문자 단위(charater-level)의 경계 상자(bounding box)만 결정되면 region score map과 affinity score map을 구할 수 있다.

위와 같은 방식으로 정의된 ground-truth는 문자를 개별적인 요소로 다룬다. 따라서 가로로 아주 긴 단어나 사이즈가 큰 텍스트를 비교적 작은 receptive field로도 잘 감지할 수 있다는 장점이 있다. 이것은 기존의 bounding box regression과 같은 접근 방식이 지닌 문제점을 보완하는 데 탁월한 것으로 보인다.
그러나, 앞에서 제시된 ground-truth를 이미지마다 하나하나 만들기에는 너무 많은 비용이 요구된다. 기존에 공개되어있는 대부분의 데이터들은 annotation이 문자 수준이 아닌 단어 수준으로 labeling되어있다. 이러한 문제를 해결하기 위한 CRAFT 연구팀에서는 기존의 단어수준 annotation이 포함된 데이터 셋을 활용하여 character-boc를 생성하는 방법을 제안했다.

### 중간 모델 학습

![](https://miro.medium.com/max/700/1*BHsihgVScwwOUg6VfGrTMA.png)

ground-truth가 포함된 학습 데이터셋을 이용해 모델을 학습하는 방법은 간단하다. 다음의 목적 함수를 최적화하면 된다.

목적함수(Loss Function) :

![](https://miro.medium.com/max/700/1*HvvYhDKWdmej_vKI5A9SvA.png)

각 픽셀 p에 대한 region score의 예측값 Sr(p)와 정답Sr*(p)의 유클리드 거리 오차와 각 픽셀 p에 대한 affinity score의 예측값 Sa(p)와 정답 Sa*(p)의 유클리드 거리 오차의 총합을 Loss로 정의하며, 이 함수를 최적화하는 것이 학습의 방향이다.

문자 수준의 레이블이 포함된 학습 데이터가 충분히 많다면 위 방식의 학습만으로도 충분할 수 있다. 그러나 앞에서 언급했듯이 대부분의 벤치마크 데이터 집합들은 단어 수준으로 레이블링되어있다.
따라서, 위 과정으로 학습이 완료된 모델은 완성된 것이 아닌 중간 단계의 모델로 Interim model이라 칭한다.

### Interim Model을 개선하는 방법

기존의 많은 단어 수준 annotation 기반 데이터 집합을 활용하기 위해 해당 데이터들이 주어졌을 때, 앞에서 학습한 interim model을 이용하여 character-box를 생성하는 방법을 소개한다.

![](https://miro.medium.com/max/700/1*X97KqXolnfz4S2abu10sDw.png)

위 이미지는 단어 수준 annotation과 함께 실제 이미지가 주어졌을 때 문자 수준 annotation을 구하기 위한 일련의 과정이다.

1. word box 단위로 이미지를 crop 한다.
2. Interim model을 이용하여 crop된 이미지의 region score를 예측한다.
3. watershed algorithm을 이용하여 문자 영역을 분리하여 character box를 결정한다.
4. 마지막으로 분리된 charactoer box들이 crop되기 이전의 원본 이미지 좌표로 이동시킨다.

단어 수준 annotation으로부터 문자 수준 annotation을 도출한 후 각 character box를 이용해 region score map과 affinity score map을 구할 수 있다.

![](https://miro.medium.com/max/700/1*hWxBV_Dq3h8Rnl_PN8etLA.png)

이 방법을 통해 얻은 ground-truth는 interim model에 의해 예측된 것으로 실제 정답과는 오차가 존재한다. 따라서 pseudo-ground-truth라 부른다.

interim model로 예측한 character box를 통해 얻은 pseudo-ground truth는 학습 시 모델의 예측 정확도에 악영향을 줄 수 있다. 따라서, 학습 과정에서는 pseudo-ground truth의 신뢰도를 반영하여 최적화한다.

#### Pseudo-Ground Truth의 신뢰도

Pseudo-GT의 신뢰도(Confidence score)는 Interim model이 cropped image에서 character box를 얼마나 잘 분리했는지 정도로 결정된다.
confidence score는 각각의 Word Box:w 마다 다음과 같이 계산된다.

![](https://miro.medium.com/max/700/1*L3EfP74mvespMDkoyaBbJQ.png)

실제 단어의 길이와 Interim model에 의해 split된 chracter box 개수 사이의 오차 비율을 계산하는 것이다.

![](https://miro.medium.com/max/700/1*jduSlZ9QaaeMypxZOrruAQ.png)

예를 들어, 'COURSE'라는 단어(L(w)=6)의 word box를 character box로 분리하는 과정에서 5개의 chracter box가 예측되었다면, confidence score는 5/6이 된다.

![](https://miro.medium.com/max/700/1*Jgw5dOlfM3eT1S9HnPm-WQ.png)

주의해야할 점은 word box의 confidence score가 0.5 보다 낮을 경우이다.
이 경우, 모델 학습 시 부정적인 영향을 줄 가능성이 매우 높기 때문에 추정된 character box를 그대로 사용하는 대신에 word box를 단어 길이 L(w)로 등분한다. 그리고 confidence score는 0.5로 설정한다. 당연히 pseudo-ground truth도 등분된 character box로 만들어낸다.

#### confidence map 생성

이미지의 각 word-box에 대해 confidence score를 계산한 후 이 값들을 이용해 다음 식을 만족하는 confidence map을 만든다.

![](https://miro.medium.com/max/700/1*7S7uCUIQjaIpIMznf2Meyw.png)

픽셀 p에 대해 p가 원본 이미지에서 word-box의 영역에 포함된다면 해당 좌표와 대응되는 confidence map의 좌표값은 word-box의 confidence score로 설정한다.
나머지 영역은 모두 1로 설정한다.

![](https://miro.medium.com/max/700/1*TyJbxiOL_Zc9qu-2Kva_7Q.png)

이렇게 만든 confidence map은 pseudo-GT 데이터셋을 학습할 때 목적함수 내에서 사용된다.

![](https://miro.medium.com/max/700/1*yzer-vF0kGizphV1873A-Q.png)

결국 word box에 대응되는 픽셀들은 interim model의 문자 영역 예측 정확도에 비례하는 가중치가 부여되는 것이다. 
학습이 진행됨에 따라 interim model은 문자를 점점 더 잘 예측하게 된다.

![](https://miro.medium.com/max/700/1*pEylC3C_xkjLXpIs_dBagw.png)

학습 과정을 요약하면, chracter level annotation 데이터셋의 부족함을 보완하기 위해 2단계의 학습을 수행한다.

1. Ground Truth Label로 1차 학습 : Interim model 생성
2. Pseudo-Ground Truth Label로 2차 학습 : Weakly-Supervised Learning

* 모델 학습 방법을 살펴봤지만 region, affinity 점수로 quadbox, polygon은 어떻게 그리나?
* 기존 방법보다 어떤 점이 우위인가?

### 개별 문자를 단어로 연결하기

앞에서는 CRAFT 모델이 이미지 내의 문자 영역을 인식할 수 있도록 학습하는 방법을 알아보았다. 마지막으로 모델을 통해 예측한 region score와 affinity score를 이용하여 최존 결과물을 도출하는 방법을 알아보자

목적에 따라 chracter box, word box, polygon 등 다양한 형태의 결과물을 만들어 낼 수 있다.

#### word level quadbox

region score와 affinity score를 통해 각 문자를 하나의 단어로 그룹화하는 방법은 다음과 같다.

1. 원본 이미지와 동일한 크기의 이진맵 M을 모든 값이 0이 되도록 초기화한다.
2. 픽셀 p에 대해 region score(p) 또는 affinity score(p)가 각각 역치값 Tr(p), Ta(p) 보다 클 경우 해당 픽셀을 1로 설정한다.
3. 이진맵 M에 대해 connected component Labeling을 수행한다.
4. 각 label을 둘러싸는 최소 영역의 회전된 직사각형을 찾는다.(OpenCV에서 제공하는 connectedComponents()와 minAreaRect()를 활용한다)

![](https://miro.medium.com/max/700/1*ASWF5Er_O8NAYsBY2lLESA.png)

#### Polygon 예측

![](https://miro.medium.com/max/700/1*_EyygIYQyQPqUk-w-OaKjw.png)

word level quadbox를 다음과 같은 과정을 통해 문자 영역 주위에 다각형을 만들어 곡선 텍스트를 효과적으로 처리할 수 있다.

1. 수직 방향을 따라 문자 영역의 local maxima line(파란선)을 찾는다.
2. 최종 다각형 결과가 불균일해지는 것을 방지하기 위해 local maxima line은 quadbox 내의 최대 길이로 통일한다.
3. local maxima line의 중심점들을 연결하는 선인 center line(노란선)을 구한다.
4. 문자 기울기 각도를 반영하기 위해 local maxima line을 center line에 수직하도록 회전한다.(빨간선)
5. local maxima line의 양 끝점은 다각형의 control point 후보들이다.
6. 두 개의 가장 바깥쪽으로 기울어진 local maxima line을 center line을 따라 바깥쪽으로 이동하여 최종 control point를 결정한다. - 텍스트 영역을 완전히 커버하기 위하여

이렇게 해서 CRAFT 모델을 학습하는 방법과 모델의 예측값을 이용해 텍스트 영역을 추출하는 방법을 살펴보았다.

### CRAFT가 다른 알고리즘보다 훌륭한 이유

전체 텍스트 대신 개별 문자를 인식하고, 상향식(bottom-up)으로 문자들을 연결하는 접근 방식
큰 이미지에서 비죠적 작은 receptive field로 하나의 문자를 감지하기에 충분하므로, 스케일이 가변적인 텍스트를 검출하는 데 있어서 견고하다.
곡선이나 기형적인 형태의 텍스트와 같은 복잡한 STD 문제에서도 높은 유연성을 보여줄 수 있다.
기존의 boundingbox regeression을 활용하는 object detection 기반 접근 방식은 하나의 경계상자로 감지하기 어려운 텍스트를 포함한 이미지에서는 어려움이 많다. 이러한 문제를 보완하기 위해 다양한 스케일의 앵커박스가 요구된다. CRAFT의 접근 방식은 여러 개의 앵커박스가 필요하지 않기 때문에 NonMaximun Suppression(NMS)과 같은 후처리 작업도 필요가 없다.

#### 한계점

* 각 문자가 분리되어있지 않고 연결되는 특정 국가의 언어 혹은 필기체에 대해서는 학습이 쉽지 않다.
* 아쉬운 FPS

CRAFT 모델은 기존의 Scene Text Detection을 위한 접근 방식들에 비해 뛰어난 결과를 보여주고 있다.


#### craft_demo

/demo_image 폴더에 있는 이미지의 텍스트를 detection하여 개별 이미지로 저장


참고]
* https://openaccess.thecvf.com/content_CVPR_2019/papers/Baek_Character_Region_Awareness_for_Text_Detection_CVPR_2019_paper.pdf
* https://github.com/clovaai/CRAFT-pytorch
* https://www.youtube.com/watch?v=HI8MzpY8KMI
* https://en.wikipedia.org/wiki/Connected-component_labeling

---
## 2. TextRecognitionDataGenerator : 인식 모델
텍스트 인식을 위한 데이터 생성기.
OCR 학습을 위해서 font와 txt 파일을 이용하여 텍스트 이미지를 생성.

### 설치
pip 패키지로 설치
> pip install trdg

source로 설치
> git clone https://github.com/Belval/TextRecognitionDataGenerator
> cd TextRecognitionDataGenerator
> pip3 install -r requirements.txt

### 사용법
* -l, --language
생성하려는 데이터에 사용할 언어 사전을 선택
* -i, --input_file
제공된 사전이 부적합할 때 사전으로 사용할 파일을 지정
* -c, --count
생성할 이미지 수, default는 1000
* -t, --thread_count
쓰레드 수. -t 8은 TRDG가 8개의 프로세스를 생성하여 작업하도록 지정
* -h, --help
아래와 같은 도움말을 표시한다.
```
usage: trdg [-h] [--output_dir [OUTPUT_DIR]] [-i [INPUT_FILE]] [-l [LANGUAGE]]
            -c [COUNT] [-rs] [-let] [-num] [-sym] [-w [LENGTH]] [-r]
            [-f [FORMAT]] [-t [THREAD_COUNT]] [-e [EXTENSION]]
            [-k [SKEW_ANGLE]] [-rk] [-wk] [-bl [BLUR]] [-rbl]
            [-b [BACKGROUND]] [-hw] [-na NAME_FORMAT] [-d [DISTORSION]]
            [-do [DISTORSION_ORIENTATION]] [-wd [WIDTH]] [-al [ALIGNMENT]]
            [-or [ORIENTATION]] [-tc [TEXT_COLOR]] [-sw [SPACE_WIDTH]]
            [-cs [CHARACTER_SPACING]] [-m [MARGINS]] [-fi] [-ft [FONT]]
            [-ca [CASE]]
Generate synthetic text data for text recognition.
optional arguments:
  -h, --help            show this help message and exit
  --output_dir [OUTPUT_DIR]
                        The output directory
  -i [INPUT_FILE], --input_file [INPUT_FILE]
                        When set, this argument uses a specified text file as
                        source for the text
  -l [LANGUAGE], --language [LANGUAGE]
                        The language to use, should be fr (French), en
                        (English), es (Spanish), de (German), or cn (Chinese).
  -c [COUNT], --count [COUNT]
                        The number of images to be created.
  -rs, --random_sequences
                        Use random sequences as the source text for the
                        generation. Set '-let','-num','-sym' to use
                        letters/numbers/symbols. If none specified, using all
                        three.
  -let, --include_letters
                        Define if random sequences should contain letters.
                        Only works with -rs
  -num, --include_numbers
                        Define if random sequences should contain numbers.
                        Only works with -rs
  -sym, --include_symbols
                        Define if random sequences should contain symbols.
                        Only works with -rs
  -w [LENGTH], --length [LENGTH]
                        Define how many words should be included in each
                        generated sample. If the text source is Wikipedia,
                        this is the MINIMUM length
  -r, --random          Define if the produced string will have variable word
                        count (with --length being the maximum)
  -f [FORMAT], --format [FORMAT]
                        Define the height of the produced images if
                        horizontal, else the width
  -t [THREAD_COUNT], --thread_count [THREAD_COUNT]
                        Define the number of thread to use for image
                        generation
  -e [EXTENSION], --extension [EXTENSION]
                        Define the extension to save the image with
  -k [SKEW_ANGLE], --skew_angle [SKEW_ANGLE]
                        Define skewing angle of the generated text. In
                        positive degrees
  -rk, --random_skew    When set, the skew angle will be randomized between
                        the value set with -k and it's opposite
  -wk, --use_wikipedia  Use Wikipedia as the source text for the generation,
                        using this paremeter ignores -r, -n, -s
  -bl [BLUR], --blur [BLUR]
                        Apply gaussian blur to the resulting sample. Should be
                        an integer defining the blur radius
  -rbl, --random_blur   When set, the blur radius will be randomized between 0
                        and -bl.
  -b [BACKGROUND], --background [BACKGROUND]
                        Define what kind of background to use. 0: Gaussian
                        Noise, 1: Plain white, 2: Quasicrystal, 3: Pictures
  -hw, --handwritten    Define if the data will be "handwritten" by an RNN
  -na NAME_FORMAT, --name_format NAME_FORMAT
                        Define how the produced files will be named. 0:
                        [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT]
                        + one file labels.txt containing id-to-label mappings
  -d [DISTORSION], --distorsion [DISTORSION]
                        Define a distorsion applied to the resulting image. 0:
                        None (Default), 1: Sine wave, 2: Cosine wave, 3:
                        Random
  -do [DISTORSION_ORIENTATION], --distorsion_orientation [DISTORSION_ORIENTATION]
                        Define the distorsion's orientation. Only used if -d
                        is specified. 0: Vertical (Up and down), 1: Horizontal
                        (Left and Right), 2: Both
  -wd [WIDTH], --width [WIDTH]
                        Define the width of the resulting image. If not set it
                        will be the width of the text + 10. If the width of
                        the generated text is bigger that number will be used
  -al [ALIGNMENT], --alignment [ALIGNMENT]
                        Define the alignment of the text in the image. Only
                        used if the width parameter is set. 0: left, 1:
                        center, 2: right
  -or [ORIENTATION], --orientation [ORIENTATION]
                        Define the orientation of the text. 0: Horizontal, 1:
                        Vertical
  -tc [TEXT_COLOR], --text_color [TEXT_COLOR]
                        Define the text's color, should be either a single hex
                        color or a range in the ?,? format.
  -sw [SPACE_WIDTH], --space_width [SPACE_WIDTH]
                        Define the width of the spaces between words. 2.0
                        means twice the normal space width
  -cs [CHARACTER_SPACING], --character_spacing [CHARACTER_SPACING]
                        Define the width of the spaces between characters. 2
                        means two pixels
  -m [MARGINS], --margins [MARGINS]
                        Define the margins around the text when rendered. In
                        pixels
  -fi, --fit            Apply a tight crop around the rendered text
  -ft [FONT], --font [FONT]
                        Define font to be used
  -ca [CASE], --case [CASE]
                        Generate upper or lowercase only. arguments: upper or
                        lower. Example: --case upper
```

### 데이터 생성
다음과 같이 실행하면, out 디렉토리에 여러 가지 폰트를 사용하여 영어 단어 이미지를 1000개 생성한다.
> python3 run.py -c 10

다음과 같이 실행하면, dicts/ko.txt 파일에 있는 사전과 fonts/ko 디렉토리에 있는 폰트를 이용하여 10개의 이미지를 생성한다.
> python3 run.py -c 10 -l ko

다음은 문자 간격없이 5개 문자로 이루어진 단어 이미지를 생성한다.
> python3 run.py -c 10 -l ko -w 5 -sw 0

보다 자세한 사용법은 사용법 페이지 -https://textrecognitiondatagenerator.readthedocs.io/en/latest/index.html - 를 참조

---
## 3. deep-text-recognition

### 데이터셋
lmdb 형태의 데이터셋이 필요하다.
lmdb 형태는 ground_truth.txt와 이미지 파일로 나누어지고, gt.txt는 이미지 파일 패스와 gt 값을 tab으로 구분하는 형식이다.

#### aihub

aihub에 공개된 "한국어글자체이미지" 데이터셋을 이용하여 학습에 활용
"한국어글자체이미지" 데이터셋은 손글씨 문자, 단어, 문장과 인쇄체 문자, 단어, 문장 그리고 책표지, 상품 이미지, 간판, 교통 표지판 등의 문자 등으로 구성되어있다.

#### TRDG

2장에 소개한 TRDG를 이용. 다양한 한글 폰트와 다양한 사전을 이용하여 학습에 필요한 단어 이미지를 생성하여 학습에 활용

#### 데이터셋 제작

1. data 폴더와 data_lmdb 폴더를 만든다.
2. data 폴더와 data_lmdb 폴더에 training, validation 폴더를 만든다.
3. training, validation 폴더에 학습, 검증용 이미지를 복사한다.
4. data 폴더에 train_gt.txt, valid_gt.txt 파일을 만든다. 

xxx_gt.txt 파일은 다음과 같은 형식으로 작성한다.
```
# 라벨링 방법
# {image_path}\t{label}\n

a.png	apple
b.png	table
c.png	bus

```

5. 이미지와 xxx_gt.txt 파일을 lmdb 로 변환

```
python create_lmdb_dataset.py --inputPath data/training --gtFile data/train_gt.txt --outputPath data_lmdb/training

python create_lmdb_dataset.py --inputPath data/validataion --gtFile data/valid_gt.txt --outputPath data_lmdb/validation
```

### 학습

#### train.py 수정

default argument 수정

* --ᅟbatch_size : default 값은 192 자신의 vram 환경에 맞게 수정해서 사용
* --select_data : default='MJ-ST'를 '/'로 수정 
* --batch_ratio : default='0.5-0.5'를 '1'로 수정
* --character : 학습할 문자
* --valinterval : default=2000, 2000 에포크마다 validation 결과가 출력

#### 모델 학습

다음 명령으로 모델을 학습시킨다.

```
python train.py --train_data data_lmdb/training --valid_data data_lmdb/validation --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn

```

학습관련 로그는 saved_models 폴더에 log_train.txt, opt.txt에 저장된다.

#### 최적화 함수 (Optimizer)
defult는 Adadelta. adam으로 변경할 수 있다.

#### Loss 함수
prediction 모듈을 CTC(Connectionist Temporal Classification)로 설정할 결우 CTCLoss라는 CTC 전용 Loss 함수를 사용한다. 이는 torch의 기본 nn 라이브러리에서 제공한다.
prediction 모듈을 Attn(Attention-based Sequence Prediction)으로 설정하면, Loss 함수는 흔히 사용하는 CrossEntropyLoss로 변경된다.

#### transformation
NONE|TPS

#### feature extraction
VGG|ResNET
이미지의 feature를 추출한다. VGG, RCNN, ResNet, GRCL(Gated RCNN) 총 4 가지가 있다.

#### sequence modeling
NONE|BiLSTM

#### prediction
CTC|Attn


### 논문 

대부분의 OCR 방법은 STR(Scene Text Recognition) 과제에서 효과적이지 못했다. 실제 세계에서 생기는 다양한 텍스트 외관과 이러한 scenes(배경)들이 포착되는 불완전한 조건 때문이다. 이런한 도전을 해결하기 위해 선행 논문에서 multi-stage(transe, feat, seq, pred 각 stage) 파이프라인을 제안했는데 여기서 각 단계는 특정 과제를 다루는 deep neural network다.

예를 들어, [24 : An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. // IEEEE, 2017]에서 입력의 다양한 문자 수를 처리하기 위해 RNN을 사용하고, 문자의 수를 식별하기 위해  Temporal Classification Loss[Connectionist temporal classification : labelling unsegmented sequence data with recurrent neural networks. // ICML, 2006]를 사용하는 방법을 제안했다. [25 : Robust scene text recognition with automatic rectification. // CVPR, 2016]에서는 downstream 모듈이 곡선 테스트를 처리하기 위한 표현 burden(부담)을 줄이기 위해 직선 텍스트 이미지로 입력을 정규화하는 변환 모듈을 제안했다.
그러나, 표1의 reported results를 보면, 학습 모델들이 서로 다른 학습 및 평가 데이터셋을 통해 나온 정확도이기 때문에, 최신 STR 모델의 개선 여부와 어떤 점에서 개선되었는지를 평가하기는 어렵다.

![](https://blog.kakaocdn.net/dn/bLxJc5/btqDnv2cepe/wbcpKH4PJFOa7xJVgekiJk/img.png)
Tabel 1. Reported Results는 일관되지 않은 학습 및 평가 데이터셋으로 표현된 기존의 STR 모델의 성능을 보여준다. 이러한 일관되지 않은 데이터셋은 STR 모델들 사이의 공정한 비교를 방해한다. Out Experiment는 통일되고 일관된 환경에서 재실험한 결과를 보여준다. 마지막 줄에는 Our Best Model은 우리가 찾아낸 최고의 모델(4단계 STR Framework)인데, 이것은 최근의 STR 모델과 비교했을 때 경쟁적인 성능을 보여준다. MJ, ST, C, PRI는 각각 MJSynth, SynthTextm, Character-labeled, 개인 데이터를 나타낸다. 각 벤치마크의 최고 정확도는 굵은 글씨로 표시된다.

우리는 기존 STR 모델에 사용했던 학습 데이터셋과 평가 데이터셋이 일치하지 않는다는 것을 관찰했다. 예를 들어, 여러 작업에서 평가 데이터셋의 일부로 IC13 데이터셋의 다른 부분 집합을 사용하므로 15% 이상의 성능 차이가 발생할 수 있다. 이러한 종류의 불일치는 다른 모델들 간의 공정한 성능 비교를 방해한다.
우리 논문은 다음과 같은 주요 contribution으로 이러한 유형(데이터셋 불일치)의 이슈를 다룬다.
첫째, STR 논문에서 흔히 사용되는 모든 train 및 evaluation 데이터셋을 분석한다. 우리의 분석은 STR 데이터셋의 사용과 그 원인에 대한 모순을 드러낸다. 예를 들어, 우리는 IC03 데이터셋에서 7개의 누락된 예와 IC13 데이터셋에서도 158개의 누락된 예제를 발견했다. 우리는 STR 데이터셋에 대한 이전 몇 가지 작업을 조사하여, 표 1과 같이 train 및 evaluation 데이터셋의 불일치가 비교 불가능한 결과를 초래한다는 것을 보여준다.
둘째, 기존 방식에 대한 공통적인 시각을 제공하는 STR의 통일 프레임워크를 도입한다. 구체적으로 STR 모델을 변환(Trans), 형상추출(Feat.), 시퀀스 모델링(Seq.) 및 예측(Pred.)의 4개지 연속 작업 단계로 구분한다. 프레임워크는 기존 방법뿐만아니라 모듈 지향적 기여도를 광범위하게 분석하기 위해 가능한 변형을 제공한다.
마지막으로, 우리는 통일된 실험 환경에서 정확성, 속도 및 메모리 소비량 측면에서 모듈에 의한 기여를 연구한다. 이 연구를 통해 개별 모듈의 기여도를 보다 엄격하게 평가하고, 이전에 간과한 모듈 조합을 제안하며, 이는 art 의 상태에 비해 개선된다. 또한 STR의 남은 과제를 파악하기 위해서 벤치마크 데이터셋의 실패 사례를 분석하였다.

......


참고]
* https://arxiv.org/pdf/1904.01906.pdf
* https://github.com/clovaai/deep-text-recognition-benchmark


#### recognition_demo

/demo_image 폴더에 있는 이미지의 텍스트를 detection하여 개별 이미지 파일로 저장하고,
개별 이미지로 저장된 detection된 텍스트를 인식하여 text_value를 예측한다.
