Ļļ
Ż£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.0-dev202007052v1.12.1-35711-g8202ae9d5c8¹

ref_enc_d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameref_enc_d1/kernel
y
%ref_enc_d1/kernel/Read/ReadVariableOpReadVariableOpref_enc_d1/kernel* 
_output_shapes
:
*
dtype0
w
ref_enc_d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameref_enc_d1/bias
p
#ref_enc_d1/bias/Read/ReadVariableOpReadVariableOpref_enc_d1/bias*
_output_shapes	
:*
dtype0

ref_z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameref_z_log_var/kernel
~
(ref_z_log_var/kernel/Read/ReadVariableOpReadVariableOpref_z_log_var/kernel*
_output_shapes
:	*
dtype0
|
ref_z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameref_z_log_var/bias
u
&ref_z_log_var/bias/Read/ReadVariableOpReadVariableOpref_z_log_var/bias*
_output_shapes
:*
dtype0

ref_z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_nameref_z_mean/kernel
x
%ref_z_mean/kernel/Read/ReadVariableOpReadVariableOpref_z_mean/kernel*
_output_shapes
:	*
dtype0
v
ref_z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameref_z_mean/bias
o
#ref_z_mean/bias/Read/ReadVariableOpReadVariableOpref_z_mean/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

decoder_d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_namedecoder_d1/kernel
x
%decoder_d1/kernel/Read/ReadVariableOpReadVariableOpdecoder_d1/kernel*
_output_shapes
:	*
dtype0
w
decoder_d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedecoder_d1/bias
p
#decoder_d1/bias/Read/ReadVariableOpReadVariableOpdecoder_d1/bias*
_output_shapes	
:*
dtype0

decoder_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_namedecoder_output/kernel

)decoder_output/kernel/Read/ReadVariableOpReadVariableOpdecoder_output/kernel* 
_output_shapes
:
*
dtype0

decoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namedecoder_output/bias
x
'decoder_output/bias/Read/ReadVariableOpReadVariableOpdecoder_output/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/ref_enc_d1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/ref_enc_d1/kernel/m

,Adam/ref_enc_d1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ref_enc_d1/kernel/m* 
_output_shapes
:
*
dtype0

Adam/ref_enc_d1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/ref_enc_d1/bias/m
~
*Adam/ref_enc_d1/bias/m/Read/ReadVariableOpReadVariableOpAdam/ref_enc_d1/bias/m*
_output_shapes	
:*
dtype0

Adam/ref_z_log_var/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameAdam/ref_z_log_var/kernel/m

/Adam/ref_z_log_var/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ref_z_log_var/kernel/m*
_output_shapes
:	*
dtype0

Adam/ref_z_log_var/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/ref_z_log_var/bias/m

-Adam/ref_z_log_var/bias/m/Read/ReadVariableOpReadVariableOpAdam/ref_z_log_var/bias/m*
_output_shapes
:*
dtype0

Adam/ref_z_mean/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/ref_z_mean/kernel/m

,Adam/ref_z_mean/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ref_z_mean/kernel/m*
_output_shapes
:	*
dtype0

Adam/ref_z_mean/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/ref_z_mean/bias/m
}
*Adam/ref_z_mean/bias/m/Read/ReadVariableOpReadVariableOpAdam/ref_z_mean/bias/m*
_output_shapes
:*
dtype0

Adam/decoder_d1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/decoder_d1/kernel/m

,Adam/decoder_d1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoder_d1/kernel/m*
_output_shapes
:	*
dtype0

Adam/decoder_d1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/decoder_d1/bias/m
~
*Adam/decoder_d1/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoder_d1/bias/m*
_output_shapes	
:*
dtype0

Adam/decoder_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameAdam/decoder_output/kernel/m

0Adam/decoder_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoder_output/kernel/m* 
_output_shapes
:
*
dtype0

Adam/decoder_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/decoder_output/bias/m

.Adam/decoder_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoder_output/bias/m*
_output_shapes	
:*
dtype0

Adam/ref_enc_d1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/ref_enc_d1/kernel/v

,Adam/ref_enc_d1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ref_enc_d1/kernel/v* 
_output_shapes
:
*
dtype0

Adam/ref_enc_d1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/ref_enc_d1/bias/v
~
*Adam/ref_enc_d1/bias/v/Read/ReadVariableOpReadVariableOpAdam/ref_enc_d1/bias/v*
_output_shapes	
:*
dtype0

Adam/ref_z_log_var/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameAdam/ref_z_log_var/kernel/v

/Adam/ref_z_log_var/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ref_z_log_var/kernel/v*
_output_shapes
:	*
dtype0

Adam/ref_z_log_var/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/ref_z_log_var/bias/v

-Adam/ref_z_log_var/bias/v/Read/ReadVariableOpReadVariableOpAdam/ref_z_log_var/bias/v*
_output_shapes
:*
dtype0

Adam/ref_z_mean/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/ref_z_mean/kernel/v

,Adam/ref_z_mean/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ref_z_mean/kernel/v*
_output_shapes
:	*
dtype0

Adam/ref_z_mean/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/ref_z_mean/bias/v
}
*Adam/ref_z_mean/bias/v/Read/ReadVariableOpReadVariableOpAdam/ref_z_mean/bias/v*
_output_shapes
:*
dtype0

Adam/decoder_d1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/decoder_d1/kernel/v

,Adam/decoder_d1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoder_d1/kernel/v*
_output_shapes
:	*
dtype0

Adam/decoder_d1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/decoder_d1/bias/v
~
*Adam/decoder_d1/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoder_d1/bias/v*
_output_shapes	
:*
dtype0

Adam/decoder_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameAdam/decoder_output/kernel/v

0Adam/decoder_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoder_output/kernel/v* 
_output_shapes
:
*
dtype0

Adam/decoder_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/decoder_output/bias/v

.Adam/decoder_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoder_output/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
ĖK
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*K
valueüJBłJ BņJ
ß
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
	optimizer
loss
regularization_losses
	variables
trainable_variables
 	keras_api
!
signatures
 
ī
layer-0
layer_with_weights-0
layer-1
layer-2
	layer_with_weights-1
	layer-3
layer_with_weights-2
layer-4
"layer-5
#regularization_losses
$	variables
%trainable_variables
&	keras_api
ŗ
'layer-0
(layer_with_weights-0
(layer-1
)layer-2
*layer_with_weights-1
*layer-3
+regularization_losses
,	variables
-trainable_variables
.	keras_api
h

/kernel
0bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
R
5regularization_losses
6	variables
7trainable_variables
8	keras_api

9	keras_api

:	keras_api
h

;kernel
<bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
h

Akernel
Bbias
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api

G	keras_api

H	keras_api

I	keras_api

J	keras_api

K	keras_api

L	keras_api

M	keras_api

N	keras_api

O	keras_api

P	keras_api

Q	keras_api

R	keras_api

S	keras_api

T	keras_api

U	keras_api

V	keras_api
R
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api

[iter

\beta_1

]beta_2
	^decay
_learning_rate/mµ0m¶;m·<møAm¹Bmŗ`m»am¼bm½cm¾/væ0vĄ;vĮ<vĀAvĆBvÄ`vÅavĘbvĒcvČ
 
 
F
/0
01
A2
B3
;4
<5
`6
a7
b8
c9
F
/0
01
A2
B3
;4
<5
`6
a7
b8
c9
­
dnon_trainable_variables
elayer_regularization_losses
regularization_losses
flayer_metrics
gmetrics
	variables
trainable_variables

hlayers
 
R
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
 
*
/0
01
A2
B3
;4
<5
*
/0
01
A2
B3
;4
<5
­
mnon_trainable_variables
nlayer_regularization_losses
#regularization_losses
olayer_metrics
pmetrics
$	variables
%trainable_variables

qlayers
 
h

`kernel
abias
rregularization_losses
s	variables
ttrainable_variables
u	keras_api
R
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
h

bkernel
cbias
zregularization_losses
{	variables
|trainable_variables
}	keras_api
 

`0
a1
b2
c3

`0
a1
b2
c3
°
~non_trainable_variables
layer_regularization_losses
+regularization_losses
layer_metrics
metrics
,	variables
-trainable_variables
layers
][
VARIABLE_VALUEref_enc_d1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEref_enc_d1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
²
non_trainable_variables
 layer_regularization_losses
1regularization_losses
layer_metrics
metrics
2	variables
3trainable_variables
layers
 
 
 
²
non_trainable_variables
 layer_regularization_losses
5regularization_losses
layer_metrics
metrics
6	variables
7trainable_variables
layers
 
 
`^
VARIABLE_VALUEref_z_log_var/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEref_z_log_var/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

;0
<1
²
non_trainable_variables
 layer_regularization_losses
=regularization_losses
layer_metrics
metrics
>	variables
?trainable_variables
layers
][
VARIABLE_VALUEref_z_mean/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEref_z_mean/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

A0
B1

A0
B1
²
non_trainable_variables
 layer_regularization_losses
Cregularization_losses
layer_metrics
metrics
D	variables
Etrainable_variables
layers
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
²
non_trainable_variables
 layer_regularization_losses
Wregularization_losses
layer_metrics
metrics
X	variables
Ytrainable_variables
layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdecoder_d1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdecoder_d1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEdecoder_output/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdecoder_output/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
Ę
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
 
 
 
²
non_trainable_variables
 layer_regularization_losses
iregularization_losses
layer_metrics
 metrics
j	variables
ktrainable_variables
”layers
 
 
 
 
*
0
1
2
	3
4
"5
 

`0
a1

`0
a1
²
¢non_trainable_variables
 £layer_regularization_losses
rregularization_losses
¤layer_metrics
„metrics
s	variables
ttrainable_variables
¦layers
 
 
 
²
§non_trainable_variables
 Ølayer_regularization_losses
vregularization_losses
©layer_metrics
Ŗmetrics
w	variables
xtrainable_variables
«layers
 

b0
c1

b0
c1
²
¬non_trainable_variables
 ­layer_regularization_losses
zregularization_losses
®layer_metrics
Æmetrics
{	variables
|trainable_variables
°layers
 
 
 
 

'0
(1
)2
*3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

±total

²count
³	variables
“	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

±0
²1

³	variables
~
VARIABLE_VALUEAdam/ref_enc_d1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/ref_enc_d1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/ref_z_log_var/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/ref_z_log_var/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/ref_z_mean/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/ref_z_mean/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/decoder_d1/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/decoder_d1/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/decoder_output/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/decoder_output/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/ref_enc_d1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/ref_enc_d1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/ref_z_log_var/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/ref_z_log_var/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/ref_z_mean/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/ref_z_mean/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/decoder_d1/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/decoder_d1/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/decoder_output/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/decoder_output/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_encoder_inputPlaceholder*(
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’

StatefulPartitionedCallStatefulPartitionedCallserving_default_encoder_inputref_enc_d1/kernelref_enc_d1/biasref_z_mean/kernelref_z_mean/biasref_z_log_var/kernelref_z_log_var/biasdecoder_d1/kerneldecoder_d1/biasdecoder_output/kerneldecoder_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_15535
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ė
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%ref_enc_d1/kernel/Read/ReadVariableOp#ref_enc_d1/bias/Read/ReadVariableOp(ref_z_log_var/kernel/Read/ReadVariableOp&ref_z_log_var/bias/Read/ReadVariableOp%ref_z_mean/kernel/Read/ReadVariableOp#ref_z_mean/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp%decoder_d1/kernel/Read/ReadVariableOp#decoder_d1/bias/Read/ReadVariableOp)decoder_output/kernel/Read/ReadVariableOp'decoder_output/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/ref_enc_d1/kernel/m/Read/ReadVariableOp*Adam/ref_enc_d1/bias/m/Read/ReadVariableOp/Adam/ref_z_log_var/kernel/m/Read/ReadVariableOp-Adam/ref_z_log_var/bias/m/Read/ReadVariableOp,Adam/ref_z_mean/kernel/m/Read/ReadVariableOp*Adam/ref_z_mean/bias/m/Read/ReadVariableOp,Adam/decoder_d1/kernel/m/Read/ReadVariableOp*Adam/decoder_d1/bias/m/Read/ReadVariableOp0Adam/decoder_output/kernel/m/Read/ReadVariableOp.Adam/decoder_output/bias/m/Read/ReadVariableOp,Adam/ref_enc_d1/kernel/v/Read/ReadVariableOp*Adam/ref_enc_d1/bias/v/Read/ReadVariableOp/Adam/ref_z_log_var/kernel/v/Read/ReadVariableOp-Adam/ref_z_log_var/bias/v/Read/ReadVariableOp,Adam/ref_z_mean/kernel/v/Read/ReadVariableOp*Adam/ref_z_mean/bias/v/Read/ReadVariableOp,Adam/decoder_d1/kernel/v/Read/ReadVariableOp*Adam/decoder_d1/bias/v/Read/ReadVariableOp0Adam/decoder_output/kernel/v/Read/ReadVariableOp.Adam/decoder_output/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_16226
ā
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameref_enc_d1/kernelref_enc_d1/biasref_z_log_var/kernelref_z_log_var/biasref_z_mean/kernelref_z_mean/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedecoder_d1/kerneldecoder_d1/biasdecoder_output/kerneldecoder_output/biastotalcountAdam/ref_enc_d1/kernel/mAdam/ref_enc_d1/bias/mAdam/ref_z_log_var/kernel/mAdam/ref_z_log_var/bias/mAdam/ref_z_mean/kernel/mAdam/ref_z_mean/bias/mAdam/decoder_d1/kernel/mAdam/decoder_d1/bias/mAdam/decoder_output/kernel/mAdam/decoder_output/bias/mAdam/ref_enc_d1/kernel/vAdam/ref_enc_d1/bias/vAdam/ref_z_log_var/kernel/vAdam/ref_z_log_var/bias/vAdam/ref_z_mean/kernel/vAdam/ref_z_mean/bias/vAdam/decoder_d1/kernel/vAdam/decoder_d1/bias/vAdam/decoder_output/kernel/vAdam/decoder_output/bias/v*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_16347¦Ń

H
,__inference_activation_1_layer_call_fn_16072

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_149812
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
æ
”
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15078

inputs
decoder_d1_15066
decoder_d1_15068
decoder_output_15072
decoder_output_15074
identity¢"decoder_d1/StatefulPartitionedCall¢&decoder_output/StatefulPartitionedCall
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_d1_15066decoder_d1_15068*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_149602$
"decoder_d1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_149812
activation_1/PartitionedCallŅ
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_15072decoder_output_15074*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_150002(
&decoder_output/StatefulPartitionedCallŅ
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	

9__inference_vae-ref-512-2-msle-g-relu_layer_call_fn_15767

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:’’’’’’’’’: *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_154762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:’’’’’’’’’::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¤
o
C__inference_add_loss_layer_call_and_return_conditional_losses_15228

inputs
identity

identity_1I
IdentityIdentityinputs*
T0*
_output_shapes
: 2

IdentityM

Identity_1Identityinputs*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
Ó
­
E__inference_decoder_d1_layer_call_and_return_conditional_losses_16053

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

¹
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15899

inputs-
)decoder_d1_matmul_readvariableop_resource.
*decoder_d1_biasadd_readvariableop_resource1
-decoder_output_matmul_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identityÆ
 decoder_d1/MatMul/ReadVariableOpReadVariableOp)decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 decoder_d1/MatMul/ReadVariableOp
decoder_d1/MatMulMatMulinputs(decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
decoder_d1/MatMul®
!decoder_d1/BiasAdd/ReadVariableOpReadVariableOp*decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!decoder_d1/BiasAdd/ReadVariableOp®
decoder_d1/BiasAddBiasAdddecoder_d1/MatMul:product:0)decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
decoder_d1/BiasAdd~
activation_1/ReluReludecoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
activation_1/Relu¼
$decoder_output/MatMul/ReadVariableOpReadVariableOp-decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$decoder_output/MatMul/ReadVariableOpŗ
decoder_output/MatMulMatMulactivation_1/Relu:activations:0,decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
decoder_output/MatMulŗ
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%decoder_output/BiasAdd/ReadVariableOp¾
decoder_output/BiasAddBiasAdddecoder_output/MatMul:product:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
decoder_output/BiasAdd
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
decoder_output/Sigmoido
IdentityIdentitydecoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’:::::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ŗ
ź
9__inference_enc-ref-512-2-msle-g-relu_layer_call_fn_15881

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2¢StatefulPartitionedCallć
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_149272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:’’’’’’’’’::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
øQ
Ö
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15240
encoder_input#
enc_ref_512_2_msle_g_relu_15135#
enc_ref_512_2_msle_g_relu_15137#
enc_ref_512_2_msle_g_relu_15139#
enc_ref_512_2_msle_g_relu_15141#
enc_ref_512_2_msle_g_relu_15143#
enc_ref_512_2_msle_g_relu_15145#
dec_ref_512_2_msle_g_relu_15176#
dec_ref_512_2_msle_g_relu_15178#
dec_ref_512_2_msle_g_relu_15180#
dec_ref_512_2_msle_g_relu_15182
identity

identity_1¢1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall¢1enc-ref-512-2-msle-g-relu/StatefulPartitionedCall¢"ref_enc_d1/StatefulPartitionedCall¢%ref_z_log_var/StatefulPartitionedCall¢"ref_z_mean/StatefulPartitionedCall¤
1enc-ref-512-2-msle-g-relu/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_ref_512_2_msle_g_relu_15135enc_ref_512_2_msle_g_relu_15137enc_ref_512_2_msle_g_relu_15139enc_ref_512_2_msle_g_relu_15141enc_ref_512_2_msle_g_relu_15143enc_ref_512_2_msle_g_relu_15145*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_1488323
1enc-ref-512-2-msle-g-relu/StatefulPartitionedCallä
1dec-ref-512-2-msle-g-relu/StatefulPartitionedCallStatefulPartitionedCall:enc-ref-512-2-msle-g-relu/StatefulPartitionedCall:output:2dec_ref_512_2_msle_g_relu_15176dec_ref_512_2_msle_g_relu_15178dec_ref_512_2_msle_g_relu_15180dec_ref_512_2_msle_g_relu_15182*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_1505023
1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall
!tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *æÖ32#
!tf_op_layer_Maximum_1/Maximum_1/yŹ
tf_op_layer_Maximum_1/Maximum_1Maximumencoder_input*tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2!
tf_op_layer_Maximum_1/Maximum_1
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *æÖ32
tf_op_layer_Maximum/Maximum/yė
tf_op_layer_Maximum/MaximumMaximum:dec-ref-512-2-msle-g-relu/StatefulPartitionedCall:output:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Maximum/MaximumÄ
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_ref_512_2_msle_g_relu_15135enc_ref_512_2_msle_g_relu_15137*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_147252$
"ref_enc_d1/StatefulPartitionedCall
tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_1/AddV2_1/yŅ
tf_op_layer_AddV2_1/AddV2_1AddV2#tf_op_layer_Maximum_1/Maximum_1:z:0&tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_1/AddV2_1{
tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2/AddV2/yĀ
tf_op_layer_AddV2/AddV2AddV2tf_op_layer_Maximum/Maximum:z:0"tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2/AddV2
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_147462
activation/PartitionedCall
tf_op_layer_Log/LogLogtf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Log/Log
tf_op_layer_Log_1/Log_1Logtf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Log_1/Log_1ā
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0enc_ref_512_2_msle_g_relu_15143enc_ref_512_2_msle_g_relu_15145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCallŁ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0enc_ref_512_2_msle_g_relu_15139enc_ref_512_2_msle_g_relu_15141*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
"ref_z_mean/StatefulPartitionedCallļ
/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifferencetf_op_layer_Log/Log:y:0tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’21
/tf_op_layer_SquaredDifference/SquaredDifference
tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_2/AddV2_2/xÜ
tf_op_layer_AddV2_2/AddV2_2AddV2&tf_op_layer_AddV2_2/AddV2_2/x:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_2/AddV2_2®
tf_op_layer_Square/SquareSquare+ref_z_mean/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Square/Square¢
tf_op_layer_Exp/ExpExp.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Exp/Exp
'tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2)
'tf_op_layer_Mean/Mean/reduction_indicesŚ
tf_op_layer_Mean/MeanMean3tf_op_layer_SquaredDifference/SquaredDifference:z:00tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mean/Mean²
tf_op_layer_Sub/SubSubtf_op_layer_AddV2_2/AddV2_2:z:0tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sub/Subs
tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD2
tf_op_layer_Mul/Mul/y®
tf_op_layer_Mul/MulMultf_op_layer_Mean/Mean:output:0tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mul/Mul¬
tf_op_layer_Sub_1/Sub_1Subtf_op_layer_Sub/Sub:z:0tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sub_1/Sub_1
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2'
%tf_op_layer_Sum/Sum/reduction_indices»
tf_op_layer_Sum/SumSumtf_op_layer_Sub_1/Sub_1:z:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sum/Sum{
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   æ2
tf_op_layer_Mul_1/Mul_1/yø
tf_op_layer_Mul_1/Mul_1Multf_op_layer_Sum/Sum:output:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mul_1/Mul_1¶
tf_op_layer_AddV2_3/AddV2_3AddV2tf_op_layer_Mul/Mul:z:0tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_3/AddV2_3¤
+tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Mean_1/Mean_1/reduction_indicesÅ
tf_op_layer_Mean_1/Mean_1Meantf_op_layer_AddV2_3/AddV2_3:z:04tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mean_1/Mean_1ä
add_loss/PartitionedCallPartitionedCall"tf_op_layer_Mean_1/Mean_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_152282
add_loss/PartitionedCallé
IdentityIdentity:dec-ref-512-2-msle-g-relu/StatefulPartitionedCall:output:02^dec-ref-512-2-msle-g-relu/StatefulPartitionedCall2^enc-ref-512-2-msle-g-relu/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

IdentityĀ

Identity_1Identity!add_loss/PartitionedCall:output:12^dec-ref-512-2-msle-g-relu/StatefulPartitionedCall2^enc-ref-512-2-msle-g-relu/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:’’’’’’’’’::::::::::2f
1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall2f
1enc-ref-512-2-msle-g-relu/StatefulPartitionedCall1enc-ref-512-2-msle-g-relu/StatefulPartitionedCall2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:W S
(
_output_shapes
:’’’’’’’’’
'
_user_specified_nameencoder_input
Ō
°
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_14790

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¹
c
G__inference_activation_1_layer_call_and_return_conditional_losses_14981

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:’’’’’’’’’2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ė
Æ
9__inference_dec-ref-512-2-msle-g-relu_layer_call_fn_15061
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCall	latent_inunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_150502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:’’’’’’’’’
#
_user_specified_name	latent_in
Ņ
D
(__inference_add_loss_layer_call_fn_16021

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_152282
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
Q
Ļ
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15476

inputs#
enc_ref_512_2_msle_g_relu_15410#
enc_ref_512_2_msle_g_relu_15412#
enc_ref_512_2_msle_g_relu_15414#
enc_ref_512_2_msle_g_relu_15416#
enc_ref_512_2_msle_g_relu_15418#
enc_ref_512_2_msle_g_relu_15420#
dec_ref_512_2_msle_g_relu_15425#
dec_ref_512_2_msle_g_relu_15427#
dec_ref_512_2_msle_g_relu_15429#
dec_ref_512_2_msle_g_relu_15431
identity

identity_1¢1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall¢1enc-ref-512-2-msle-g-relu/StatefulPartitionedCall¢"ref_enc_d1/StatefulPartitionedCall¢%ref_z_log_var/StatefulPartitionedCall¢"ref_z_mean/StatefulPartitionedCall
1enc-ref-512-2-msle-g-relu/StatefulPartitionedCallStatefulPartitionedCallinputsenc_ref_512_2_msle_g_relu_15410enc_ref_512_2_msle_g_relu_15412enc_ref_512_2_msle_g_relu_15414enc_ref_512_2_msle_g_relu_15416enc_ref_512_2_msle_g_relu_15418enc_ref_512_2_msle_g_relu_15420*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_1492723
1enc-ref-512-2-msle-g-relu/StatefulPartitionedCallä
1dec-ref-512-2-msle-g-relu/StatefulPartitionedCallStatefulPartitionedCall:enc-ref-512-2-msle-g-relu/StatefulPartitionedCall:output:2dec_ref_512_2_msle_g_relu_15425dec_ref_512_2_msle_g_relu_15427dec_ref_512_2_msle_g_relu_15429dec_ref_512_2_msle_g_relu_15431*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_1507823
1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall
!tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *æÖ32#
!tf_op_layer_Maximum_1/Maximum_1/yĆ
tf_op_layer_Maximum_1/Maximum_1Maximuminputs*tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2!
tf_op_layer_Maximum_1/Maximum_1
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *æÖ32
tf_op_layer_Maximum/Maximum/yė
tf_op_layer_Maximum/MaximumMaximum:dec-ref-512-2-msle-g-relu/StatefulPartitionedCall:output:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Maximum/Maximum½
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_ref_512_2_msle_g_relu_15410enc_ref_512_2_msle_g_relu_15412*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_147252$
"ref_enc_d1/StatefulPartitionedCall
tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_1/AddV2_1/yŅ
tf_op_layer_AddV2_1/AddV2_1AddV2#tf_op_layer_Maximum_1/Maximum_1:z:0&tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_1/AddV2_1{
tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2/AddV2/yĀ
tf_op_layer_AddV2/AddV2AddV2tf_op_layer_Maximum/Maximum:z:0"tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2/AddV2
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_147462
activation/PartitionedCall
tf_op_layer_Log/LogLogtf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Log/Log
tf_op_layer_Log_1/Log_1Logtf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Log_1/Log_1ā
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0enc_ref_512_2_msle_g_relu_15418enc_ref_512_2_msle_g_relu_15420*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCallŁ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0enc_ref_512_2_msle_g_relu_15414enc_ref_512_2_msle_g_relu_15416*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
"ref_z_mean/StatefulPartitionedCallļ
/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifferencetf_op_layer_Log/Log:y:0tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’21
/tf_op_layer_SquaredDifference/SquaredDifference
tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_2/AddV2_2/xÜ
tf_op_layer_AddV2_2/AddV2_2AddV2&tf_op_layer_AddV2_2/AddV2_2/x:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_2/AddV2_2®
tf_op_layer_Square/SquareSquare+ref_z_mean/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Square/Square¢
tf_op_layer_Exp/ExpExp.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Exp/Exp
'tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2)
'tf_op_layer_Mean/Mean/reduction_indicesŚ
tf_op_layer_Mean/MeanMean3tf_op_layer_SquaredDifference/SquaredDifference:z:00tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mean/Mean²
tf_op_layer_Sub/SubSubtf_op_layer_AddV2_2/AddV2_2:z:0tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sub/Subs
tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD2
tf_op_layer_Mul/Mul/y®
tf_op_layer_Mul/MulMultf_op_layer_Mean/Mean:output:0tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mul/Mul¬
tf_op_layer_Sub_1/Sub_1Subtf_op_layer_Sub/Sub:z:0tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sub_1/Sub_1
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2'
%tf_op_layer_Sum/Sum/reduction_indices»
tf_op_layer_Sum/SumSumtf_op_layer_Sub_1/Sub_1:z:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sum/Sum{
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   æ2
tf_op_layer_Mul_1/Mul_1/yø
tf_op_layer_Mul_1/Mul_1Multf_op_layer_Sum/Sum:output:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mul_1/Mul_1¶
tf_op_layer_AddV2_3/AddV2_3AddV2tf_op_layer_Mul/Mul:z:0tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_3/AddV2_3¤
+tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Mean_1/Mean_1/reduction_indicesÅ
tf_op_layer_Mean_1/Mean_1Meantf_op_layer_AddV2_3/AddV2_3:z:04tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mean_1/Mean_1ä
add_loss/PartitionedCallPartitionedCall"tf_op_layer_Mean_1/Mean_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_152282
add_loss/PartitionedCallé
IdentityIdentity:dec-ref-512-2-msle-g-relu/StatefulPartitionedCall:output:02^dec-ref-512-2-msle-g-relu/StatefulPartitionedCall2^enc-ref-512-2-msle-g-relu/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

IdentityĀ

Identity_1Identity!add_loss/PartitionedCall:output:12^dec-ref-512-2-msle-g-relu/StatefulPartitionedCall2^enc-ref-512-2-msle-g-relu/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:’’’’’’’’’::::::::::2f
1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall2f
1enc-ref-512-2-msle-g-relu/StatefulPartitionedCall1enc-ref-512-2-msle-g-relu/StatefulPartitionedCall2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
P

__inference__traced_save_16226
file_prefix0
,savev2_ref_enc_d1_kernel_read_readvariableop.
*savev2_ref_enc_d1_bias_read_readvariableop3
/savev2_ref_z_log_var_kernel_read_readvariableop1
-savev2_ref_z_log_var_bias_read_readvariableop0
,savev2_ref_z_mean_kernel_read_readvariableop.
*savev2_ref_z_mean_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop0
,savev2_decoder_d1_kernel_read_readvariableop.
*savev2_decoder_d1_bias_read_readvariableop4
0savev2_decoder_output_kernel_read_readvariableop2
.savev2_decoder_output_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_ref_enc_d1_kernel_m_read_readvariableop5
1savev2_adam_ref_enc_d1_bias_m_read_readvariableop:
6savev2_adam_ref_z_log_var_kernel_m_read_readvariableop8
4savev2_adam_ref_z_log_var_bias_m_read_readvariableop7
3savev2_adam_ref_z_mean_kernel_m_read_readvariableop5
1savev2_adam_ref_z_mean_bias_m_read_readvariableop7
3savev2_adam_decoder_d1_kernel_m_read_readvariableop5
1savev2_adam_decoder_d1_bias_m_read_readvariableop;
7savev2_adam_decoder_output_kernel_m_read_readvariableop9
5savev2_adam_decoder_output_bias_m_read_readvariableop7
3savev2_adam_ref_enc_d1_kernel_v_read_readvariableop5
1savev2_adam_ref_enc_d1_bias_v_read_readvariableop:
6savev2_adam_ref_z_log_var_kernel_v_read_readvariableop8
4savev2_adam_ref_z_log_var_bias_v_read_readvariableop7
3savev2_adam_ref_z_mean_kernel_v_read_readvariableop5
1savev2_adam_ref_z_mean_bias_v_read_readvariableop7
3savev2_adam_decoder_d1_kernel_v_read_readvariableop5
1savev2_adam_decoder_d1_bias_v_read_readvariableop;
7savev2_adam_decoder_output_kernel_v_read_readvariableop9
5savev2_adam_decoder_output_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_291f08542ff84fa1a2ae11f9223ca155/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameą
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ņ
valuečBå&B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesŌ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesė
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_ref_enc_d1_kernel_read_readvariableop*savev2_ref_enc_d1_bias_read_readvariableop/savev2_ref_z_log_var_kernel_read_readvariableop-savev2_ref_z_log_var_bias_read_readvariableop,savev2_ref_z_mean_kernel_read_readvariableop*savev2_ref_z_mean_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop,savev2_decoder_d1_kernel_read_readvariableop*savev2_decoder_d1_bias_read_readvariableop0savev2_decoder_output_kernel_read_readvariableop.savev2_decoder_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_ref_enc_d1_kernel_m_read_readvariableop1savev2_adam_ref_enc_d1_bias_m_read_readvariableop6savev2_adam_ref_z_log_var_kernel_m_read_readvariableop4savev2_adam_ref_z_log_var_bias_m_read_readvariableop3savev2_adam_ref_z_mean_kernel_m_read_readvariableop1savev2_adam_ref_z_mean_bias_m_read_readvariableop3savev2_adam_decoder_d1_kernel_m_read_readvariableop1savev2_adam_decoder_d1_bias_m_read_readvariableop7savev2_adam_decoder_output_kernel_m_read_readvariableop5savev2_adam_decoder_output_bias_m_read_readvariableop3savev2_adam_ref_enc_d1_kernel_v_read_readvariableop1savev2_adam_ref_enc_d1_bias_v_read_readvariableop6savev2_adam_ref_z_log_var_kernel_v_read_readvariableop4savev2_adam_ref_z_log_var_bias_v_read_readvariableop3savev2_adam_ref_z_mean_kernel_v_read_readvariableop1savev2_adam_ref_z_mean_bias_v_read_readvariableop3savev2_adam_decoder_d1_kernel_v_read_readvariableop1savev2_adam_decoder_d1_bias_v_read_readvariableop7savev2_adam_decoder_output_kernel_v_read_readvariableop5savev2_adam_decoder_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	2
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*µ
_input_shapes£
 : :
::	::	:: : : : : :	::
:: : :
::	::	::	::
::
::	::	::	::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::% !

_output_shapes
:	: !

_output_shapes
::%"!

_output_shapes
:	:!#

_output_shapes	
::&$"
 
_output_shapes
:
:!%

_output_shapes	
::&

_output_shapes
: 
	

9__inference_vae-ref-512-2-msle-g-relu_layer_call_fn_15500
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallś
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:’’’’’’’’’: *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_154762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:’’’’’’’’’::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:’’’’’’’’’
'
_user_specified_nameencoder_input
ź

-__inference_ref_z_log_var_layer_call_fn_15991

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallū
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ļ
ń
9__inference_enc-ref-512-2-msle-g-relu_layer_call_fn_14902
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2¢StatefulPartitionedCallź
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_148832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:’’’’’’’’’::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:’’’’’’’’’
'
_user_specified_nameencoder_input
Ō
°
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_15982

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¹
c
G__inference_activation_1_layer_call_and_return_conditional_losses_16067

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:’’’’’’’’’2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	

9__inference_vae-ref-512-2-msle-g-relu_layer_call_fn_15405
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallś
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:’’’’’’’’’: *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_153812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:’’’’’’’’’::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:’’’’’’’’’
'
_user_specified_nameencoder_input
æ
”
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15050

inputs
decoder_d1_15038
decoder_d1_15040
decoder_output_15044
decoder_output_15046
identity¢"decoder_d1/StatefulPartitionedCall¢&decoder_output/StatefulPartitionedCall
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_d1_15038decoder_d1_15040*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_149602$
"decoder_d1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_149812
activation_1/PartitionedCallŅ
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_15044decoder_output_15046*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_150002(
&decoder_output/StatefulPartitionedCallŅ
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

¹
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15917

inputs-
)decoder_d1_matmul_readvariableop_resource.
*decoder_d1_biasadd_readvariableop_resource1
-decoder_output_matmul_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identityÆ
 decoder_d1/MatMul/ReadVariableOpReadVariableOp)decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 decoder_d1/MatMul/ReadVariableOp
decoder_d1/MatMulMatMulinputs(decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
decoder_d1/MatMul®
!decoder_d1/BiasAdd/ReadVariableOpReadVariableOp*decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!decoder_d1/BiasAdd/ReadVariableOp®
decoder_d1/BiasAddBiasAdddecoder_d1/MatMul:product:0)decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
decoder_d1/BiasAdd~
activation_1/ReluReludecoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
activation_1/Relu¼
$decoder_output/MatMul/ReadVariableOpReadVariableOp-decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$decoder_output/MatMul/ReadVariableOpŗ
decoder_output/MatMulMatMulactivation_1/Relu:activations:0,decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
decoder_output/MatMulŗ
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%decoder_output/BiasAdd/ReadVariableOp¾
decoder_output/BiasAddBiasAdddecoder_output/MatMul:product:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
decoder_output/BiasAdd
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
decoder_output/Sigmoido
IdentityIdentitydecoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’:::::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ō
š
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15625

inputsG
Cenc_ref_512_2_msle_g_relu_ref_enc_d1_matmul_readvariableop_resourceH
Denc_ref_512_2_msle_g_relu_ref_enc_d1_biasadd_readvariableop_resourceG
Cenc_ref_512_2_msle_g_relu_ref_z_mean_matmul_readvariableop_resourceH
Denc_ref_512_2_msle_g_relu_ref_z_mean_biasadd_readvariableop_resourceJ
Fenc_ref_512_2_msle_g_relu_ref_z_log_var_matmul_readvariableop_resourceK
Genc_ref_512_2_msle_g_relu_ref_z_log_var_biasadd_readvariableop_resourceG
Cdec_ref_512_2_msle_g_relu_decoder_d1_matmul_readvariableop_resourceH
Ddec_ref_512_2_msle_g_relu_decoder_d1_biasadd_readvariableop_resourceK
Gdec_ref_512_2_msle_g_relu_decoder_output_matmul_readvariableop_resourceL
Hdec_ref_512_2_msle_g_relu_decoder_output_biasadd_readvariableop_resource
identity

identity_1ž
:enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMul/ReadVariableOpReadVariableOpCenc_ref_512_2_msle_g_relu_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02<
:enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMul/ReadVariableOpć
+enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMulMatMulinputsBenc-ref-512-2-msle-g-relu/ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2-
+enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMulü
;enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpDenc_ref_512_2_msle_g_relu_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd/ReadVariableOp
,enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAddBiasAdd5enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMul:product:0Cenc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2.
,enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAddČ
)enc-ref-512-2-msle-g-relu/activation/ReluRelu5enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2+
)enc-ref-512-2-msle-g-relu/activation/Reluż
:enc-ref-512-2-msle-g-relu/ref_z_mean/MatMul/ReadVariableOpReadVariableOpCenc_ref_512_2_msle_g_relu_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02<
:enc-ref-512-2-msle-g-relu/ref_z_mean/MatMul/ReadVariableOp
+enc-ref-512-2-msle-g-relu/ref_z_mean/MatMulMatMul7enc-ref-512-2-msle-g-relu/activation/Relu:activations:0Benc-ref-512-2-msle-g-relu/ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2-
+enc-ref-512-2-msle-g-relu/ref_z_mean/MatMulū
;enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpDenc_ref_512_2_msle_g_relu_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd/ReadVariableOp
,enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAddBiasAdd5enc-ref-512-2-msle-g-relu/ref_z_mean/MatMul:product:0Cenc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2.
,enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd
=enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMul/ReadVariableOpReadVariableOpFenc_ref_512_2_msle_g_relu_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02?
=enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMul/ReadVariableOp
.enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMulMatMul7enc-ref-512-2-msle-g-relu/activation/Relu:activations:0Eenc-ref-512-2-msle-g-relu/ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’20
.enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMul
>enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpGenc_ref_512_2_msle_g_relu_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd/ReadVariableOp”
/enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAddBiasAdd8enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMul:product:0Fenc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’21
/enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd³
%enc-ref-512-2-msle-g-relu/ref_z/ShapeShape5enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2'
%enc-ref-512-2-msle-g-relu/ref_z/Shape­
2enc-ref-512-2-msle-g-relu/ref_z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    24
2enc-ref-512-2-msle-g-relu/ref_z/random_normal/mean±
4enc-ref-512-2-msle-g-relu/ref_z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4enc-ref-512-2-msle-g-relu/ref_z/random_normal/stddev
Benc-ref-512-2-msle-g-relu/ref_z/random_normal/RandomStandardNormalRandomStandardNormal.enc-ref-512-2-msle-g-relu/ref_z/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype02D
Benc-ref-512-2-msle-g-relu/ref_z/random_normal/RandomStandardNormal«
1enc-ref-512-2-msle-g-relu/ref_z/random_normal/mulMulKenc-ref-512-2-msle-g-relu/ref_z/random_normal/RandomStandardNormal:output:0=enc-ref-512-2-msle-g-relu/ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:’’’’’’’’’23
1enc-ref-512-2-msle-g-relu/ref_z/random_normal/mul
-enc-ref-512-2-msle-g-relu/ref_z/random_normalAdd5enc-ref-512-2-msle-g-relu/ref_z/random_normal/mul:z:0;enc-ref-512-2-msle-g-relu/ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:’’’’’’’’’2/
-enc-ref-512-2-msle-g-relu/ref_z/random_normal
%enc-ref-512-2-msle-g-relu/ref_z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%enc-ref-512-2-msle-g-relu/ref_z/mul/xķ
#enc-ref-512-2-msle-g-relu/ref_z/mulMul.enc-ref-512-2-msle-g-relu/ref_z/mul/x:output:08enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2%
#enc-ref-512-2-msle-g-relu/ref_z/mul¬
#enc-ref-512-2-msle-g-relu/ref_z/ExpExp'enc-ref-512-2-msle-g-relu/ref_z/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2%
#enc-ref-512-2-msle-g-relu/ref_z/Expć
%enc-ref-512-2-msle-g-relu/ref_z/mul_1Mul'enc-ref-512-2-msle-g-relu/ref_z/Exp:y:01enc-ref-512-2-msle-g-relu/ref_z/random_normal:z:0*
T0*'
_output_shapes
:’’’’’’’’’2'
%enc-ref-512-2-msle-g-relu/ref_z/mul_1ē
#enc-ref-512-2-msle-g-relu/ref_z/addAddV25enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd:output:0)enc-ref-512-2-msle-g-relu/ref_z/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2%
#enc-ref-512-2-msle-g-relu/ref_z/addż
:dec-ref-512-2-msle-g-relu/decoder_d1/MatMul/ReadVariableOpReadVariableOpCdec_ref_512_2_msle_g_relu_decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02<
:dec-ref-512-2-msle-g-relu/decoder_d1/MatMul/ReadVariableOp
+dec-ref-512-2-msle-g-relu/decoder_d1/MatMulMatMul'enc-ref-512-2-msle-g-relu/ref_z/add:z:0Bdec-ref-512-2-msle-g-relu/decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2-
+dec-ref-512-2-msle-g-relu/decoder_d1/MatMulü
;dec-ref-512-2-msle-g-relu/decoder_d1/BiasAdd/ReadVariableOpReadVariableOpDdec_ref_512_2_msle_g_relu_decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;dec-ref-512-2-msle-g-relu/decoder_d1/BiasAdd/ReadVariableOp
,dec-ref-512-2-msle-g-relu/decoder_d1/BiasAddBiasAdd5dec-ref-512-2-msle-g-relu/decoder_d1/MatMul:product:0Cdec-ref-512-2-msle-g-relu/decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2.
,dec-ref-512-2-msle-g-relu/decoder_d1/BiasAddĢ
+dec-ref-512-2-msle-g-relu/activation_1/ReluRelu5dec-ref-512-2-msle-g-relu/decoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2-
+dec-ref-512-2-msle-g-relu/activation_1/Relu
>dec-ref-512-2-msle-g-relu/decoder_output/MatMul/ReadVariableOpReadVariableOpGdec_ref_512_2_msle_g_relu_decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02@
>dec-ref-512-2-msle-g-relu/decoder_output/MatMul/ReadVariableOp¢
/dec-ref-512-2-msle-g-relu/decoder_output/MatMulMatMul9dec-ref-512-2-msle-g-relu/activation_1/Relu:activations:0Fdec-ref-512-2-msle-g-relu/decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’21
/dec-ref-512-2-msle-g-relu/decoder_output/MatMul
?dec-ref-512-2-msle-g-relu/decoder_output/BiasAdd/ReadVariableOpReadVariableOpHdec_ref_512_2_msle_g_relu_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02A
?dec-ref-512-2-msle-g-relu/decoder_output/BiasAdd/ReadVariableOp¦
0dec-ref-512-2-msle-g-relu/decoder_output/BiasAddBiasAdd9dec-ref-512-2-msle-g-relu/decoder_output/MatMul:product:0Gdec-ref-512-2-msle-g-relu/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’22
0dec-ref-512-2-msle-g-relu/decoder_output/BiasAddŻ
0dec-ref-512-2-msle-g-relu/decoder_output/SigmoidSigmoid9dec-ref-512-2-msle-g-relu/decoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’22
0dec-ref-512-2-msle-g-relu/decoder_output/Sigmoid
!tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *æÖ32#
!tf_op_layer_Maximum_1/Maximum_1/yĆ
tf_op_layer_Maximum_1/Maximum_1Maximuminputs*tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2!
tf_op_layer_Maximum_1/Maximum_1
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *æÖ32
tf_op_layer_Maximum/Maximum/yå
tf_op_layer_Maximum/MaximumMaximum4dec-ref-512-2-msle-g-relu/decoder_output/Sigmoid:y:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Maximum/MaximumŹ
 ref_enc_d1/MatMul/ReadVariableOpReadVariableOpCenc_ref_512_2_msle_g_relu_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 ref_enc_d1/MatMul/ReadVariableOp
ref_enc_d1/MatMulMatMulinputs(ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
ref_enc_d1/MatMulČ
!ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpDenc_ref_512_2_msle_g_relu_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ref_enc_d1/BiasAdd/ReadVariableOp®
ref_enc_d1/BiasAddBiasAddref_enc_d1/MatMul:product:0)ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
ref_enc_d1/BiasAdd
tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_1/AddV2_1/yŅ
tf_op_layer_AddV2_1/AddV2_1AddV2#tf_op_layer_Maximum_1/Maximum_1:z:0&tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_1/AddV2_1{
tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2/AddV2/yĀ
tf_op_layer_AddV2/AddV2AddV2tf_op_layer_Maximum/Maximum:z:0"tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2/AddV2z
activation/ReluReluref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
activation/Relu
tf_op_layer_Log/LogLogtf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Log/Log
tf_op_layer_Log_1/Log_1Logtf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Log_1/Log_1Ņ
#ref_z_log_var/MatMul/ReadVariableOpReadVariableOpFenc_ref_512_2_msle_g_relu_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#ref_z_log_var/MatMul/ReadVariableOp“
ref_z_log_var/MatMulMatMulactivation/Relu:activations:0+ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_log_var/MatMulŠ
$ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpGenc_ref_512_2_msle_g_relu_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ref_z_log_var/BiasAdd/ReadVariableOp¹
ref_z_log_var/BiasAddBiasAddref_z_log_var/MatMul:product:0,ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_log_var/BiasAddÉ
 ref_z_mean/MatMul/ReadVariableOpReadVariableOpCenc_ref_512_2_msle_g_relu_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ref_z_mean/MatMul/ReadVariableOp«
ref_z_mean/MatMulMatMulactivation/Relu:activations:0(ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_mean/MatMulĒ
!ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpDenc_ref_512_2_msle_g_relu_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ref_z_mean/BiasAdd/ReadVariableOp­
ref_z_mean/BiasAddBiasAddref_z_mean/MatMul:product:0)ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_mean/BiasAddļ
/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifferencetf_op_layer_Log/Log:y:0tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’21
/tf_op_layer_SquaredDifference/SquaredDifference
tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_2/AddV2_2/xĢ
tf_op_layer_AddV2_2/AddV2_2AddV2&tf_op_layer_AddV2_2/AddV2_2/x:output:0ref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_2/AddV2_2
tf_op_layer_Square/SquareSquareref_z_mean/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Square/Square
tf_op_layer_Exp/ExpExpref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Exp/Exp
'tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2)
'tf_op_layer_Mean/Mean/reduction_indicesŚ
tf_op_layer_Mean/MeanMean3tf_op_layer_SquaredDifference/SquaredDifference:z:00tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mean/Mean²
tf_op_layer_Sub/SubSubtf_op_layer_AddV2_2/AddV2_2:z:0tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sub/Subs
tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD2
tf_op_layer_Mul/Mul/y®
tf_op_layer_Mul/MulMultf_op_layer_Mean/Mean:output:0tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mul/Mul¬
tf_op_layer_Sub_1/Sub_1Subtf_op_layer_Sub/Sub:z:0tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sub_1/Sub_1
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2'
%tf_op_layer_Sum/Sum/reduction_indices»
tf_op_layer_Sum/SumSumtf_op_layer_Sub_1/Sub_1:z:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sum/Sum{
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   æ2
tf_op_layer_Mul_1/Mul_1/yø
tf_op_layer_Mul_1/Mul_1Multf_op_layer_Sum/Sum:output:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mul_1/Mul_1¶
tf_op_layer_AddV2_3/AddV2_3AddV2tf_op_layer_Mul/Mul:z:0tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_3/AddV2_3¤
+tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Mean_1/Mean_1/reduction_indicesÅ
tf_op_layer_Mean_1/Mean_1Meantf_op_layer_AddV2_3/AddV2_3:z:04tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mean_1/Mean_1
IdentityIdentity4dec-ref-512-2-msle-g-relu/decoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identityi

Identity_1Identity"tf_op_layer_Mean_1/Mean_1:output:0*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:’’’’’’’’’:::::::::::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ö
­
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_14725

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ŗ
ź
9__inference_enc-ref-512-2-msle-g-relu_layer_call_fn_15860

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2¢StatefulPartitionedCallć
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_148832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:’’’’’’’’’::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ó
­
E__inference_decoder_d1_layer_call_and_return_conditional_losses_14960

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:::O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ć

*__inference_decoder_d1_layer_call_fn_16062

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallł
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_149602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Q
Ļ
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15381

inputs#
enc_ref_512_2_msle_g_relu_15315#
enc_ref_512_2_msle_g_relu_15317#
enc_ref_512_2_msle_g_relu_15319#
enc_ref_512_2_msle_g_relu_15321#
enc_ref_512_2_msle_g_relu_15323#
enc_ref_512_2_msle_g_relu_15325#
dec_ref_512_2_msle_g_relu_15330#
dec_ref_512_2_msle_g_relu_15332#
dec_ref_512_2_msle_g_relu_15334#
dec_ref_512_2_msle_g_relu_15336
identity

identity_1¢1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall¢1enc-ref-512-2-msle-g-relu/StatefulPartitionedCall¢"ref_enc_d1/StatefulPartitionedCall¢%ref_z_log_var/StatefulPartitionedCall¢"ref_z_mean/StatefulPartitionedCall
1enc-ref-512-2-msle-g-relu/StatefulPartitionedCallStatefulPartitionedCallinputsenc_ref_512_2_msle_g_relu_15315enc_ref_512_2_msle_g_relu_15317enc_ref_512_2_msle_g_relu_15319enc_ref_512_2_msle_g_relu_15321enc_ref_512_2_msle_g_relu_15323enc_ref_512_2_msle_g_relu_15325*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_1488323
1enc-ref-512-2-msle-g-relu/StatefulPartitionedCallä
1dec-ref-512-2-msle-g-relu/StatefulPartitionedCallStatefulPartitionedCall:enc-ref-512-2-msle-g-relu/StatefulPartitionedCall:output:2dec_ref_512_2_msle_g_relu_15330dec_ref_512_2_msle_g_relu_15332dec_ref_512_2_msle_g_relu_15334dec_ref_512_2_msle_g_relu_15336*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_1505023
1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall
!tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *æÖ32#
!tf_op_layer_Maximum_1/Maximum_1/yĆ
tf_op_layer_Maximum_1/Maximum_1Maximuminputs*tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2!
tf_op_layer_Maximum_1/Maximum_1
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *æÖ32
tf_op_layer_Maximum/Maximum/yė
tf_op_layer_Maximum/MaximumMaximum:dec-ref-512-2-msle-g-relu/StatefulPartitionedCall:output:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Maximum/Maximum½
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_ref_512_2_msle_g_relu_15315enc_ref_512_2_msle_g_relu_15317*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_147252$
"ref_enc_d1/StatefulPartitionedCall
tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_1/AddV2_1/yŅ
tf_op_layer_AddV2_1/AddV2_1AddV2#tf_op_layer_Maximum_1/Maximum_1:z:0&tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_1/AddV2_1{
tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2/AddV2/yĀ
tf_op_layer_AddV2/AddV2AddV2tf_op_layer_Maximum/Maximum:z:0"tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2/AddV2
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_147462
activation/PartitionedCall
tf_op_layer_Log/LogLogtf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Log/Log
tf_op_layer_Log_1/Log_1Logtf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Log_1/Log_1ā
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0enc_ref_512_2_msle_g_relu_15323enc_ref_512_2_msle_g_relu_15325*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCallŁ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0enc_ref_512_2_msle_g_relu_15319enc_ref_512_2_msle_g_relu_15321*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
"ref_z_mean/StatefulPartitionedCallļ
/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifferencetf_op_layer_Log/Log:y:0tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’21
/tf_op_layer_SquaredDifference/SquaredDifference
tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_2/AddV2_2/xÜ
tf_op_layer_AddV2_2/AddV2_2AddV2&tf_op_layer_AddV2_2/AddV2_2/x:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_2/AddV2_2®
tf_op_layer_Square/SquareSquare+ref_z_mean/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Square/Square¢
tf_op_layer_Exp/ExpExp.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Exp/Exp
'tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2)
'tf_op_layer_Mean/Mean/reduction_indicesŚ
tf_op_layer_Mean/MeanMean3tf_op_layer_SquaredDifference/SquaredDifference:z:00tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mean/Mean²
tf_op_layer_Sub/SubSubtf_op_layer_AddV2_2/AddV2_2:z:0tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sub/Subs
tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD2
tf_op_layer_Mul/Mul/y®
tf_op_layer_Mul/MulMultf_op_layer_Mean/Mean:output:0tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mul/Mul¬
tf_op_layer_Sub_1/Sub_1Subtf_op_layer_Sub/Sub:z:0tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sub_1/Sub_1
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2'
%tf_op_layer_Sum/Sum/reduction_indices»
tf_op_layer_Sum/SumSumtf_op_layer_Sub_1/Sub_1:z:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sum/Sum{
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   æ2
tf_op_layer_Mul_1/Mul_1/yø
tf_op_layer_Mul_1/Mul_1Multf_op_layer_Sum/Sum:output:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mul_1/Mul_1¶
tf_op_layer_AddV2_3/AddV2_3AddV2tf_op_layer_Mul/Mul:z:0tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_3/AddV2_3¤
+tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Mean_1/Mean_1/reduction_indicesÅ
tf_op_layer_Mean_1/Mean_1Meantf_op_layer_AddV2_3/AddV2_3:z:04tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mean_1/Mean_1ä
add_loss/PartitionedCallPartitionedCall"tf_op_layer_Mean_1/Mean_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_152282
add_loss/PartitionedCallé
IdentityIdentity:dec-ref-512-2-msle-g-relu/StatefulPartitionedCall:output:02^dec-ref-512-2-msle-g-relu/StatefulPartitionedCall2^enc-ref-512-2-msle-g-relu/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

IdentityĀ

Identity_1Identity!add_loss/PartitionedCall:output:12^dec-ref-512-2-msle-g-relu/StatefulPartitionedCall2^enc-ref-512-2-msle-g-relu/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:’’’’’’’’’::::::::::2f
1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall2f
1enc-ref-512-2-msle-g-relu/StatefulPartitionedCall1enc-ref-512-2-msle-g-relu/StatefulPartitionedCall2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
©%
¶
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15839

inputs-
)ref_enc_d1_matmul_readvariableop_resource.
*ref_enc_d1_biasadd_readvariableop_resource-
)ref_z_mean_matmul_readvariableop_resource.
*ref_z_mean_biasadd_readvariableop_resource0
,ref_z_log_var_matmul_readvariableop_resource1
-ref_z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2°
 ref_enc_d1/MatMul/ReadVariableOpReadVariableOp)ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 ref_enc_d1/MatMul/ReadVariableOp
ref_enc_d1/MatMulMatMulinputs(ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
ref_enc_d1/MatMul®
!ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOp*ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ref_enc_d1/BiasAdd/ReadVariableOp®
ref_enc_d1/BiasAddBiasAddref_enc_d1/MatMul:product:0)ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
ref_enc_d1/BiasAddz
activation/ReluReluref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
activation/ReluÆ
 ref_z_mean/MatMul/ReadVariableOpReadVariableOp)ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ref_z_mean/MatMul/ReadVariableOp«
ref_z_mean/MatMulMatMulactivation/Relu:activations:0(ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_mean/MatMul­
!ref_z_mean/BiasAdd/ReadVariableOpReadVariableOp*ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ref_z_mean/BiasAdd/ReadVariableOp­
ref_z_mean/BiasAddBiasAddref_z_mean/MatMul:product:0)ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_mean/BiasAddø
#ref_z_log_var/MatMul/ReadVariableOpReadVariableOp,ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#ref_z_log_var/MatMul/ReadVariableOp“
ref_z_log_var/MatMulMatMulactivation/Relu:activations:0+ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_log_var/MatMul¶
$ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOp-ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ref_z_log_var/BiasAdd/ReadVariableOp¹
ref_z_log_var/BiasAddBiasAddref_z_log_var/MatMul:product:0,ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_log_var/BiasAdde
ref_z/ShapeShaperef_z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
ref_z/Shapey
ref_z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ref_z/random_normal/mean}
ref_z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ref_z/random_normal/stddevĮ
(ref_z/random_normal/RandomStandardNormalRandomStandardNormalref_z/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype02*
(ref_z/random_normal/RandomStandardNormalĆ
ref_z/random_normal/mulMul1ref_z/random_normal/RandomStandardNormal:output:0#ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z/random_normal/mul£
ref_z/random_normalAddref_z/random_normal/mul:z:0!ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z/random_normal_
ref_z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
ref_z/mul/x
	ref_z/mulMulref_z/mul/x:output:0ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
	ref_z/mul^
	ref_z/ExpExpref_z/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
	ref_z/Exp{
ref_z/mul_1Mulref_z/Exp:y:0ref_z/random_normal:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z/mul_1
	ref_z/addAddV2ref_z_mean/BiasAdd:output:0ref_z/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
	ref_z/addo
IdentityIdentityref_z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identityv

Identity_1Identityref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1e

Identity_2Identityref_z/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:’’’’’’’’’:::::::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

ū
!__inference__traced_restore_16347
file_prefix&
"assignvariableop_ref_enc_d1_kernel&
"assignvariableop_1_ref_enc_d1_bias+
'assignvariableop_2_ref_z_log_var_kernel)
%assignvariableop_3_ref_z_log_var_bias(
$assignvariableop_4_ref_z_mean_kernel&
"assignvariableop_5_ref_z_mean_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate)
%assignvariableop_11_decoder_d1_kernel'
#assignvariableop_12_decoder_d1_bias-
)assignvariableop_13_decoder_output_kernel+
'assignvariableop_14_decoder_output_bias
assignvariableop_15_total
assignvariableop_16_count0
,assignvariableop_17_adam_ref_enc_d1_kernel_m.
*assignvariableop_18_adam_ref_enc_d1_bias_m3
/assignvariableop_19_adam_ref_z_log_var_kernel_m1
-assignvariableop_20_adam_ref_z_log_var_bias_m0
,assignvariableop_21_adam_ref_z_mean_kernel_m.
*assignvariableop_22_adam_ref_z_mean_bias_m0
,assignvariableop_23_adam_decoder_d1_kernel_m.
*assignvariableop_24_adam_decoder_d1_bias_m4
0assignvariableop_25_adam_decoder_output_kernel_m2
.assignvariableop_26_adam_decoder_output_bias_m0
,assignvariableop_27_adam_ref_enc_d1_kernel_v.
*assignvariableop_28_adam_ref_enc_d1_bias_v3
/assignvariableop_29_adam_ref_z_log_var_kernel_v1
-assignvariableop_30_adam_ref_z_log_var_bias_v0
,assignvariableop_31_adam_ref_z_mean_kernel_v.
*assignvariableop_32_adam_ref_z_mean_bias_v0
,assignvariableop_33_adam_decoder_d1_kernel_v.
*assignvariableop_34_adam_decoder_d1_bias_v4
0assignvariableop_35_adam_decoder_output_kernel_v2
.assignvariableop_36_adam_decoder_output_bias_v
identity_38¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ę
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ņ
valuečBå&B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesŚ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesģ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*®
_output_shapes
::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity”
AssignVariableOpAssignVariableOp"assignvariableop_ref_enc_d1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_ref_enc_d1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¬
AssignVariableOp_2AssignVariableOp'assignvariableop_2_ref_z_log_var_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ŗ
AssignVariableOp_3AssignVariableOp%assignvariableop_3_ref_z_log_var_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4©
AssignVariableOp_4AssignVariableOp$assignvariableop_4_ref_z_mean_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOp"assignvariableop_5_ref_z_mean_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6”
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¢
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11­
AssignVariableOp_11AssignVariableOp%assignvariableop_11_decoder_d1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_decoder_d1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13±
AssignVariableOp_13AssignVariableOp)assignvariableop_13_decoder_output_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Æ
AssignVariableOp_14AssignVariableOp'assignvariableop_14_decoder_output_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15”
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16”
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17“
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_ref_enc_d1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18²
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_ref_enc_d1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19·
AssignVariableOp_19AssignVariableOp/assignvariableop_19_adam_ref_z_log_var_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20µ
AssignVariableOp_20AssignVariableOp-assignvariableop_20_adam_ref_z_log_var_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21“
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_ref_z_mean_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22²
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_ref_z_mean_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23“
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_decoder_d1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24²
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_decoder_d1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25ø
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_decoder_output_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¶
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_decoder_output_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27“
AssignVariableOp_27AssignVariableOp,assignvariableop_27_adam_ref_enc_d1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28²
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_ref_enc_d1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29·
AssignVariableOp_29AssignVariableOp/assignvariableop_29_adam_ref_z_log_var_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30µ
AssignVariableOp_30AssignVariableOp-assignvariableop_30_adam_ref_z_log_var_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31“
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_ref_z_mean_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32²
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_ref_z_mean_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33“
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_decoder_d1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34²
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_decoder_d1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35ø
AssignVariableOp_35AssignVariableOp0assignvariableop_35_adam_decoder_output_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¶
AssignVariableOp_36AssignVariableOp.assignvariableop_36_adam_decoder_output_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37’
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_38"#
identity_38Identity_38:output:0*«
_input_shapes
: :::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¹
±
I__inference_decoder_output_layer_call_and_return_conditional_losses_16083

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoid`
IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

¶
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_14857
encoder_input
ref_enc_d1_14837
ref_enc_d1_14839
ref_z_mean_14843
ref_z_mean_14845
ref_z_log_var_14848
ref_z_log_var_14850
identity

identity_1

identity_2¢"ref_enc_d1/StatefulPartitionedCall¢ref_z/StatefulPartitionedCall¢%ref_z_log_var/StatefulPartitionedCall¢"ref_z_mean/StatefulPartitionedCall¦
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputref_enc_d1_14837ref_enc_d1_14839*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_147252$
"ref_enc_d1/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_147462
activation/PartitionedCall»
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_14843ref_z_mean_14845*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
"ref_z_mean/StatefulPartitionedCallŹ
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_14848ref_z_log_var_14850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCall»
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_148222
ref_z/StatefulPartitionedCall
IdentityIdentity+ref_z_mean/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1

Identity_2Identity&ref_z/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:’’’’’’’’’::::::2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2>
ref_z/StatefulPartitionedCallref_z/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:W S
(
_output_shapes
:’’’’’’’’’
'
_user_specified_nameencoder_input
Ā
¬
9__inference_dec-ref-512-2-msle-g-relu_layer_call_fn_15943

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_150782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
·
a
E__inference_activation_layer_call_and_return_conditional_losses_15967

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:’’’’’’’’’2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Æ·
·
 __inference__wrapped_model_14711
encoder_inputa
]vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_enc_d1_matmul_readvariableop_resourceb
^vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_enc_d1_biasadd_readvariableop_resourcea
]vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_z_mean_matmul_readvariableop_resourceb
^vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_z_mean_biasadd_readvariableop_resourced
`vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_z_log_var_matmul_readvariableop_resourcee
avae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_z_log_var_biasadd_readvariableop_resourcea
]vae_ref_512_2_msle_g_relu_dec_ref_512_2_msle_g_relu_decoder_d1_matmul_readvariableop_resourceb
^vae_ref_512_2_msle_g_relu_dec_ref_512_2_msle_g_relu_decoder_d1_biasadd_readvariableop_resourcee
avae_ref_512_2_msle_g_relu_dec_ref_512_2_msle_g_relu_decoder_output_matmul_readvariableop_resourcef
bvae_ref_512_2_msle_g_relu_dec_ref_512_2_msle_g_relu_decoder_output_biasadd_readvariableop_resource
identityĢ
Tvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMul/ReadVariableOpReadVariableOp]vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02V
Tvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMul/ReadVariableOpø
Evae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMulMatMulencoder_input\vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2G
Evae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMulŹ
Uvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOp^vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02W
Uvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd/ReadVariableOpž
Fvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAddBiasAddOvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMul:product:0]vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2H
Fvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd
Cvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/activation/ReluReluOvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2E
Cvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/activation/ReluĖ
Tvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_mean/MatMul/ReadVariableOpReadVariableOp]vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02V
Tvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_mean/MatMul/ReadVariableOpū
Evae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_mean/MatMulMatMulQvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/activation/Relu:activations:0\vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2G
Evae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_mean/MatMulÉ
Uvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd/ReadVariableOpReadVariableOp^vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02W
Uvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd/ReadVariableOpż
Fvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAddBiasAddOvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_mean/MatMul:product:0]vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2H
Fvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAddŌ
Wvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMul/ReadVariableOpReadVariableOp`vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02Y
Wvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMul/ReadVariableOp
Hvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMulMatMulQvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/activation/Relu:activations:0_vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2J
Hvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMulŅ
Xvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpavae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02Z
Xvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd/ReadVariableOp
Ivae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAddBiasAddRvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMul:product:0`vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2K
Ivae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd
?vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/ShapeShapeOvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2A
?vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/Shapeį
Lvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2N
Lvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normal/meanå
Nvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2P
Nvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normal/stddevŻ
\vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normal/RandomStandardNormalRandomStandardNormalHvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype02^
\vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normal/RandomStandardNormal
Kvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normal/mulMulevae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normal/RandomStandardNormal:output:0Wvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:’’’’’’’’’2M
Kvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normal/muló
Gvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normalAddOvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normal/mul:z:0Uvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:’’’’’’’’’2I
Gvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normalĒ
?vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2A
?vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/mul/xÕ
=vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/mulMulHvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/mul/x:output:0Rvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2?
=vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/mulś
=vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/ExpExpAvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2?
=vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/ExpĖ
?vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/mul_1MulAvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/Exp:y:0Kvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/random_normal:z:0*
T0*'
_output_shapes
:’’’’’’’’’2A
?vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/mul_1Ļ
=vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/addAddV2Ovae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd:output:0Cvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2?
=vae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/addĖ
Tvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_d1/MatMul/ReadVariableOpReadVariableOp]vae_ref_512_2_msle_g_relu_dec_ref_512_2_msle_g_relu_decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02V
Tvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_d1/MatMul/ReadVariableOpģ
Evae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_d1/MatMulMatMulAvae-ref-512-2-msle-g-relu/enc-ref-512-2-msle-g-relu/ref_z/add:z:0\vae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2G
Evae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_d1/MatMulŹ
Uvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_d1/BiasAdd/ReadVariableOpReadVariableOp^vae_ref_512_2_msle_g_relu_dec_ref_512_2_msle_g_relu_decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02W
Uvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_d1/BiasAdd/ReadVariableOpž
Fvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_d1/BiasAddBiasAddOvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_d1/MatMul:product:0]vae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2H
Fvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_d1/BiasAdd
Evae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/activation_1/ReluReluOvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2G
Evae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/activation_1/ReluŲ
Xvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/MatMul/ReadVariableOpReadVariableOpavae_ref_512_2_msle_g_relu_dec_ref_512_2_msle_g_relu_decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02Z
Xvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/MatMul/ReadVariableOp
Ivae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/MatMulMatMulSvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/activation_1/Relu:activations:0`vae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2K
Ivae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/MatMulÖ
Yvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/BiasAdd/ReadVariableOpReadVariableOpbvae_ref_512_2_msle_g_relu_dec_ref_512_2_msle_g_relu_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02[
Yvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/BiasAdd/ReadVariableOp
Jvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/BiasAddBiasAddSvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/MatMul:product:0avae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2L
Jvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/BiasAdd«
Jvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/SigmoidSigmoidSvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2L
Jvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/Sigmoidæ
;vae-ref-512-2-msle-g-relu/tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *æÖ32=
;vae-ref-512-2-msle-g-relu/tf_op_layer_Maximum_1/Maximum_1/y
9vae-ref-512-2-msle-g-relu/tf_op_layer_Maximum_1/Maximum_1Maximumencoder_inputDvae-ref-512-2-msle-g-relu/tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2;
9vae-ref-512-2-msle-g-relu/tf_op_layer_Maximum_1/Maximum_1·
7vae-ref-512-2-msle-g-relu/tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *æÖ329
7vae-ref-512-2-msle-g-relu/tf_op_layer_Maximum/Maximum/yĶ
5vae-ref-512-2-msle-g-relu/tf_op_layer_Maximum/MaximumMaximumNvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/Sigmoid:y:0@vae-ref-512-2-msle-g-relu/tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’27
5vae-ref-512-2-msle-g-relu/tf_op_layer_Maximum/Maximum
:vae-ref-512-2-msle-g-relu/ref_enc_d1/MatMul/ReadVariableOpReadVariableOp]vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02<
:vae-ref-512-2-msle-g-relu/ref_enc_d1/MatMul/ReadVariableOpź
+vae-ref-512-2-msle-g-relu/ref_enc_d1/MatMulMatMulencoder_inputBvae-ref-512-2-msle-g-relu/ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2-
+vae-ref-512-2-msle-g-relu/ref_enc_d1/MatMul
;vae-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOp^vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;vae-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd/ReadVariableOp
,vae-ref-512-2-msle-g-relu/ref_enc_d1/BiasAddBiasAdd5vae-ref-512-2-msle-g-relu/ref_enc_d1/MatMul:product:0Cvae-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2.
,vae-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd·
7vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?29
7vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_1/AddV2_1/yŗ
5vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_1/AddV2_1AddV2=vae-ref-512-2-msle-g-relu/tf_op_layer_Maximum_1/Maximum_1:z:0@vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’27
5vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_1/AddV2_1Æ
3vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2/AddV2/yŖ
1vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2/AddV2AddV29vae-ref-512-2-msle-g-relu/tf_op_layer_Maximum/Maximum:z:0<vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’23
1vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2/AddV2Č
)vae-ref-512-2-msle-g-relu/activation/ReluRelu5vae-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2+
)vae-ref-512-2-msle-g-relu/activation/ReluŽ
-vae-ref-512-2-msle-g-relu/tf_op_layer_Log/LogLog5vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2/
-vae-ref-512-2-msle-g-relu/tf_op_layer_Log/Logź
1vae-ref-512-2-msle-g-relu/tf_op_layer_Log_1/Log_1Log9vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’23
1vae-ref-512-2-msle-g-relu/tf_op_layer_Log_1/Log_1 
=vae-ref-512-2-msle-g-relu/ref_z_log_var/MatMul/ReadVariableOpReadVariableOp`vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02?
=vae-ref-512-2-msle-g-relu/ref_z_log_var/MatMul/ReadVariableOp
.vae-ref-512-2-msle-g-relu/ref_z_log_var/MatMulMatMul7vae-ref-512-2-msle-g-relu/activation/Relu:activations:0Evae-ref-512-2-msle-g-relu/ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’20
.vae-ref-512-2-msle-g-relu/ref_z_log_var/MatMul
>vae-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpavae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>vae-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd/ReadVariableOp”
/vae-ref-512-2-msle-g-relu/ref_z_log_var/BiasAddBiasAdd8vae-ref-512-2-msle-g-relu/ref_z_log_var/MatMul:product:0Fvae-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’21
/vae-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd
:vae-ref-512-2-msle-g-relu/ref_z_mean/MatMul/ReadVariableOpReadVariableOp]vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02<
:vae-ref-512-2-msle-g-relu/ref_z_mean/MatMul/ReadVariableOp
+vae-ref-512-2-msle-g-relu/ref_z_mean/MatMulMatMul7vae-ref-512-2-msle-g-relu/activation/Relu:activations:0Bvae-ref-512-2-msle-g-relu/ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2-
+vae-ref-512-2-msle-g-relu/ref_z_mean/MatMul
;vae-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd/ReadVariableOpReadVariableOp^vae_ref_512_2_msle_g_relu_enc_ref_512_2_msle_g_relu_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;vae-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd/ReadVariableOp
,vae-ref-512-2-msle-g-relu/ref_z_mean/BiasAddBiasAdd5vae-ref-512-2-msle-g-relu/ref_z_mean/MatMul:product:0Cvae-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2.
,vae-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd×
Ivae-ref-512-2-msle-g-relu/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifference1vae-ref-512-2-msle-g-relu/tf_op_layer_Log/Log:y:05vae-ref-512-2-msle-g-relu/tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2K
Ivae-ref-512-2-msle-g-relu/tf_op_layer_SquaredDifference/SquaredDifference·
7vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?29
7vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_2/AddV2_2/x“
5vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_2/AddV2_2AddV2@vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_2/AddV2_2/x:output:08vae-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’27
5vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_2/AddV2_2ģ
3vae-ref-512-2-msle-g-relu/tf_op_layer_Square/SquareSquare5vae-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’25
3vae-ref-512-2-msle-g-relu/tf_op_layer_Square/Squareą
-vae-ref-512-2-msle-g-relu/tf_op_layer_Exp/ExpExp8vae-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2/
-vae-ref-512-2-msle-g-relu/tf_op_layer_Exp/ExpŃ
Avae-ref-512-2-msle-g-relu/tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2C
Avae-ref-512-2-msle-g-relu/tf_op_layer_Mean/Mean/reduction_indicesĀ
/vae-ref-512-2-msle-g-relu/tf_op_layer_Mean/MeanMeanMvae-ref-512-2-msle-g-relu/tf_op_layer_SquaredDifference/SquaredDifference:z:0Jvae-ref-512-2-msle-g-relu/tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’21
/vae-ref-512-2-msle-g-relu/tf_op_layer_Mean/Mean
-vae-ref-512-2-msle-g-relu/tf_op_layer_Sub/SubSub9vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_2/AddV2_2:z:07vae-ref-512-2-msle-g-relu/tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2/
-vae-ref-512-2-msle-g-relu/tf_op_layer_Sub/Sub§
/vae-ref-512-2-msle-g-relu/tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD21
/vae-ref-512-2-msle-g-relu/tf_op_layer_Mul/Mul/y
-vae-ref-512-2-msle-g-relu/tf_op_layer_Mul/MulMul8vae-ref-512-2-msle-g-relu/tf_op_layer_Mean/Mean:output:08vae-ref-512-2-msle-g-relu/tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2/
-vae-ref-512-2-msle-g-relu/tf_op_layer_Mul/Mul
1vae-ref-512-2-msle-g-relu/tf_op_layer_Sub_1/Sub_1Sub1vae-ref-512-2-msle-g-relu/tf_op_layer_Sub/Sub:z:01vae-ref-512-2-msle-g-relu/tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’23
1vae-ref-512-2-msle-g-relu/tf_op_layer_Sub_1/Sub_1Ķ
?vae-ref-512-2-msle-g-relu/tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2A
?vae-ref-512-2-msle-g-relu/tf_op_layer_Sum/Sum/reduction_indices£
-vae-ref-512-2-msle-g-relu/tf_op_layer_Sum/SumSum5vae-ref-512-2-msle-g-relu/tf_op_layer_Sub_1/Sub_1:z:0Hvae-ref-512-2-msle-g-relu/tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2/
-vae-ref-512-2-msle-g-relu/tf_op_layer_Sum/SumÆ
3vae-ref-512-2-msle-g-relu/tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   æ25
3vae-ref-512-2-msle-g-relu/tf_op_layer_Mul_1/Mul_1/y 
1vae-ref-512-2-msle-g-relu/tf_op_layer_Mul_1/Mul_1Mul6vae-ref-512-2-msle-g-relu/tf_op_layer_Sum/Sum:output:0<vae-ref-512-2-msle-g-relu/tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’23
1vae-ref-512-2-msle-g-relu/tf_op_layer_Mul_1/Mul_1
5vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_3/AddV2_3AddV21vae-ref-512-2-msle-g-relu/tf_op_layer_Mul/Mul:z:05vae-ref-512-2-msle-g-relu/tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’27
5vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_3/AddV2_3Ų
Evae-ref-512-2-msle-g-relu/tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2G
Evae-ref-512-2-msle-g-relu/tf_op_layer_Mean_1/Mean_1/reduction_indices­
3vae-ref-512-2-msle-g-relu/tf_op_layer_Mean_1/Mean_1Mean9vae-ref-512-2-msle-g-relu/tf_op_layer_AddV2_3/AddV2_3:z:0Nvae-ref-512-2-msle-g-relu/tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 25
3vae-ref-512-2-msle-g-relu/tf_op_layer_Mean_1/Mean_1£
IdentityIdentityNvae-ref-512-2-msle-g-relu/dec-ref-512-2-msle-g-relu/decoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:’’’’’’’’’:::::::::::W S
(
_output_shapes
:’’’’’’’’’
'
_user_specified_nameencoder_input
Ö
­
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_15953

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

F
*__inference_activation_layer_call_fn_15972

inputs
identityĒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_147462
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
©%
¶
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15803

inputs-
)ref_enc_d1_matmul_readvariableop_resource.
*ref_enc_d1_biasadd_readvariableop_resource-
)ref_z_mean_matmul_readvariableop_resource.
*ref_z_mean_biasadd_readvariableop_resource0
,ref_z_log_var_matmul_readvariableop_resource1
-ref_z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2°
 ref_enc_d1/MatMul/ReadVariableOpReadVariableOp)ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 ref_enc_d1/MatMul/ReadVariableOp
ref_enc_d1/MatMulMatMulinputs(ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
ref_enc_d1/MatMul®
!ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOp*ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ref_enc_d1/BiasAdd/ReadVariableOp®
ref_enc_d1/BiasAddBiasAddref_enc_d1/MatMul:product:0)ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
ref_enc_d1/BiasAddz
activation/ReluReluref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
activation/ReluÆ
 ref_z_mean/MatMul/ReadVariableOpReadVariableOp)ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ref_z_mean/MatMul/ReadVariableOp«
ref_z_mean/MatMulMatMulactivation/Relu:activations:0(ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_mean/MatMul­
!ref_z_mean/BiasAdd/ReadVariableOpReadVariableOp*ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ref_z_mean/BiasAdd/ReadVariableOp­
ref_z_mean/BiasAddBiasAddref_z_mean/MatMul:product:0)ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_mean/BiasAddø
#ref_z_log_var/MatMul/ReadVariableOpReadVariableOp,ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#ref_z_log_var/MatMul/ReadVariableOp“
ref_z_log_var/MatMulMatMulactivation/Relu:activations:0+ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_log_var/MatMul¶
$ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOp-ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ref_z_log_var/BiasAdd/ReadVariableOp¹
ref_z_log_var/BiasAddBiasAddref_z_log_var/MatMul:product:0,ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_log_var/BiasAdde
ref_z/ShapeShaperef_z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
ref_z/Shapey
ref_z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
ref_z/random_normal/mean}
ref_z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
ref_z/random_normal/stddevĮ
(ref_z/random_normal/RandomStandardNormalRandomStandardNormalref_z/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype02*
(ref_z/random_normal/RandomStandardNormalĆ
ref_z/random_normal/mulMul1ref_z/random_normal/RandomStandardNormal:output:0#ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z/random_normal/mul£
ref_z/random_normalAddref_z/random_normal/mul:z:0!ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z/random_normal_
ref_z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
ref_z/mul/x
	ref_z/mulMulref_z/mul/x:output:0ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
	ref_z/mul^
	ref_z/ExpExpref_z/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
	ref_z/Exp{
ref_z/mul_1Mulref_z/Exp:y:0ref_z/random_normal:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z/mul_1
	ref_z/addAddV2ref_z_mean/BiasAdd:output:0ref_z/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
	ref_z/addo
IdentityIdentityref_z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identityv

Identity_1Identityref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1e

Identity_2Identityref_z/add:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:’’’’’’’’’:::::::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

¶
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_14834
encoder_input
ref_enc_d1_14736
ref_enc_d1_14738
ref_z_mean_14775
ref_z_mean_14777
ref_z_log_var_14801
ref_z_log_var_14803
identity

identity_1

identity_2¢"ref_enc_d1/StatefulPartitionedCall¢ref_z/StatefulPartitionedCall¢%ref_z_log_var/StatefulPartitionedCall¢"ref_z_mean/StatefulPartitionedCall¦
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputref_enc_d1_14736ref_enc_d1_14738*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_147252$
"ref_enc_d1/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_147462
activation/PartitionedCall»
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_14775ref_z_mean_14777*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
"ref_z_mean/StatefulPartitionedCallŹ
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_14801ref_z_log_var_14803*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCall»
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_148222
ref_z/StatefulPartitionedCall
IdentityIdentity+ref_z_mean/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1

Identity_2Identity&ref_z/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:’’’’’’’’’::::::2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2>
ref_z/StatefulPartitionedCallref_z/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:W S
(
_output_shapes
:’’’’’’’’’
'
_user_specified_nameencoder_input
Č
¤
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15017
	latent_in
decoder_d1_14971
decoder_d1_14973
decoder_output_15011
decoder_output_15013
identity¢"decoder_d1/StatefulPartitionedCall¢&decoder_output/StatefulPartitionedCall¢
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCall	latent_indecoder_d1_14971decoder_d1_14973*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_149602$
"decoder_d1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_149812
activation_1/PartitionedCallŅ
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_15011decoder_output_15013*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_150002(
&decoder_output/StatefulPartitionedCallŅ
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:R N
'
_output_shapes
:’’’’’’’’’
#
_user_specified_name	latent_in
·
a
E__inference_activation_layer_call_and_return_conditional_losses_14746

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:’’’’’’’’’2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*'
_input_shapes
:’’’’’’’’’:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
©
o
@__inference_ref_z_layer_call_and_return_conditional_losses_16037
inputs_0
inputs_1
identityF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_normal/stddevÆ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype02$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:’’’’’’’’’2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1

n
%__inference_ref_z_layer_call_fn_16043
inputs_0
inputs_1
identity¢StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_148222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs/1

Æ
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_14927

inputs
ref_enc_d1_14907
ref_enc_d1_14909
ref_z_mean_14913
ref_z_mean_14915
ref_z_log_var_14918
ref_z_log_var_14920
identity

identity_1

identity_2¢"ref_enc_d1/StatefulPartitionedCall¢ref_z/StatefulPartitionedCall¢%ref_z_log_var/StatefulPartitionedCall¢"ref_z_mean/StatefulPartitionedCall
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallinputsref_enc_d1_14907ref_enc_d1_14909*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_147252$
"ref_enc_d1/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_147462
activation/PartitionedCall»
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_14913ref_z_mean_14915*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
"ref_z_mean/StatefulPartitionedCallŹ
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_14918ref_z_log_var_14920*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCall»
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_148222
ref_z/StatefulPartitionedCall
IdentityIdentity+ref_z_mean/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1

Identity_2Identity&ref_z/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:’’’’’’’’’::::::2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2>
ref_z/StatefulPartitionedCallref_z/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ć

*__inference_ref_z_mean_layer_call_fn_16010

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallų
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Č
¤
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15032
	latent_in
decoder_d1_15020
decoder_d1_15022
decoder_output_15026
decoder_output_15028
identity¢"decoder_d1/StatefulPartitionedCall¢&decoder_output/StatefulPartitionedCall¢
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCall	latent_indecoder_d1_15020decoder_d1_15022*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_149602$
"decoder_d1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_149812
activation_1/PartitionedCallŅ
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_15026decoder_output_15028*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_150002(
&decoder_output/StatefulPartitionedCallŅ
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:R N
'
_output_shapes
:’’’’’’’’’
#
_user_specified_name	latent_in

Æ
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_14883

inputs
ref_enc_d1_14863
ref_enc_d1_14865
ref_z_mean_14869
ref_z_mean_14871
ref_z_log_var_14874
ref_z_log_var_14876
identity

identity_1

identity_2¢"ref_enc_d1/StatefulPartitionedCall¢ref_z/StatefulPartitionedCall¢%ref_z_log_var/StatefulPartitionedCall¢"ref_z_mean/StatefulPartitionedCall
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallinputsref_enc_d1_14863ref_enc_d1_14865*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_147252$
"ref_enc_d1/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_147462
activation/PartitionedCall»
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_14869ref_z_mean_14871*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
"ref_z_mean/StatefulPartitionedCallŹ
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_14874ref_z_log_var_14876*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCall»
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_148222
ref_z/StatefulPartitionedCall
IdentityIdentity+ref_z_mean/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1

Identity_2Identity&ref_z/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:’’’’’’’’’::::::2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2>
ref_z/StatefulPartitionedCallref_z/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ō
š
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15715

inputsG
Cenc_ref_512_2_msle_g_relu_ref_enc_d1_matmul_readvariableop_resourceH
Denc_ref_512_2_msle_g_relu_ref_enc_d1_biasadd_readvariableop_resourceG
Cenc_ref_512_2_msle_g_relu_ref_z_mean_matmul_readvariableop_resourceH
Denc_ref_512_2_msle_g_relu_ref_z_mean_biasadd_readvariableop_resourceJ
Fenc_ref_512_2_msle_g_relu_ref_z_log_var_matmul_readvariableop_resourceK
Genc_ref_512_2_msle_g_relu_ref_z_log_var_biasadd_readvariableop_resourceG
Cdec_ref_512_2_msle_g_relu_decoder_d1_matmul_readvariableop_resourceH
Ddec_ref_512_2_msle_g_relu_decoder_d1_biasadd_readvariableop_resourceK
Gdec_ref_512_2_msle_g_relu_decoder_output_matmul_readvariableop_resourceL
Hdec_ref_512_2_msle_g_relu_decoder_output_biasadd_readvariableop_resource
identity

identity_1ž
:enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMul/ReadVariableOpReadVariableOpCenc_ref_512_2_msle_g_relu_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02<
:enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMul/ReadVariableOpć
+enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMulMatMulinputsBenc-ref-512-2-msle-g-relu/ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2-
+enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMulü
;enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpDenc_ref_512_2_msle_g_relu_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd/ReadVariableOp
,enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAddBiasAdd5enc-ref-512-2-msle-g-relu/ref_enc_d1/MatMul:product:0Cenc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2.
,enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAddČ
)enc-ref-512-2-msle-g-relu/activation/ReluRelu5enc-ref-512-2-msle-g-relu/ref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2+
)enc-ref-512-2-msle-g-relu/activation/Reluż
:enc-ref-512-2-msle-g-relu/ref_z_mean/MatMul/ReadVariableOpReadVariableOpCenc_ref_512_2_msle_g_relu_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02<
:enc-ref-512-2-msle-g-relu/ref_z_mean/MatMul/ReadVariableOp
+enc-ref-512-2-msle-g-relu/ref_z_mean/MatMulMatMul7enc-ref-512-2-msle-g-relu/activation/Relu:activations:0Benc-ref-512-2-msle-g-relu/ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2-
+enc-ref-512-2-msle-g-relu/ref_z_mean/MatMulū
;enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpDenc_ref_512_2_msle_g_relu_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd/ReadVariableOp
,enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAddBiasAdd5enc-ref-512-2-msle-g-relu/ref_z_mean/MatMul:product:0Cenc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2.
,enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd
=enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMul/ReadVariableOpReadVariableOpFenc_ref_512_2_msle_g_relu_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02?
=enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMul/ReadVariableOp
.enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMulMatMul7enc-ref-512-2-msle-g-relu/activation/Relu:activations:0Eenc-ref-512-2-msle-g-relu/ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’20
.enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMul
>enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpGenc_ref_512_2_msle_g_relu_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd/ReadVariableOp”
/enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAddBiasAdd8enc-ref-512-2-msle-g-relu/ref_z_log_var/MatMul:product:0Fenc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’21
/enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd³
%enc-ref-512-2-msle-g-relu/ref_z/ShapeShape5enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2'
%enc-ref-512-2-msle-g-relu/ref_z/Shape­
2enc-ref-512-2-msle-g-relu/ref_z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    24
2enc-ref-512-2-msle-g-relu/ref_z/random_normal/mean±
4enc-ref-512-2-msle-g-relu/ref_z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4enc-ref-512-2-msle-g-relu/ref_z/random_normal/stddev
Benc-ref-512-2-msle-g-relu/ref_z/random_normal/RandomStandardNormalRandomStandardNormal.enc-ref-512-2-msle-g-relu/ref_z/Shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype02D
Benc-ref-512-2-msle-g-relu/ref_z/random_normal/RandomStandardNormal«
1enc-ref-512-2-msle-g-relu/ref_z/random_normal/mulMulKenc-ref-512-2-msle-g-relu/ref_z/random_normal/RandomStandardNormal:output:0=enc-ref-512-2-msle-g-relu/ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:’’’’’’’’’23
1enc-ref-512-2-msle-g-relu/ref_z/random_normal/mul
-enc-ref-512-2-msle-g-relu/ref_z/random_normalAdd5enc-ref-512-2-msle-g-relu/ref_z/random_normal/mul:z:0;enc-ref-512-2-msle-g-relu/ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:’’’’’’’’’2/
-enc-ref-512-2-msle-g-relu/ref_z/random_normal
%enc-ref-512-2-msle-g-relu/ref_z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%enc-ref-512-2-msle-g-relu/ref_z/mul/xķ
#enc-ref-512-2-msle-g-relu/ref_z/mulMul.enc-ref-512-2-msle-g-relu/ref_z/mul/x:output:08enc-ref-512-2-msle-g-relu/ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2%
#enc-ref-512-2-msle-g-relu/ref_z/mul¬
#enc-ref-512-2-msle-g-relu/ref_z/ExpExp'enc-ref-512-2-msle-g-relu/ref_z/mul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2%
#enc-ref-512-2-msle-g-relu/ref_z/Expć
%enc-ref-512-2-msle-g-relu/ref_z/mul_1Mul'enc-ref-512-2-msle-g-relu/ref_z/Exp:y:01enc-ref-512-2-msle-g-relu/ref_z/random_normal:z:0*
T0*'
_output_shapes
:’’’’’’’’’2'
%enc-ref-512-2-msle-g-relu/ref_z/mul_1ē
#enc-ref-512-2-msle-g-relu/ref_z/addAddV25enc-ref-512-2-msle-g-relu/ref_z_mean/BiasAdd:output:0)enc-ref-512-2-msle-g-relu/ref_z/mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2%
#enc-ref-512-2-msle-g-relu/ref_z/addż
:dec-ref-512-2-msle-g-relu/decoder_d1/MatMul/ReadVariableOpReadVariableOpCdec_ref_512_2_msle_g_relu_decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02<
:dec-ref-512-2-msle-g-relu/decoder_d1/MatMul/ReadVariableOp
+dec-ref-512-2-msle-g-relu/decoder_d1/MatMulMatMul'enc-ref-512-2-msle-g-relu/ref_z/add:z:0Bdec-ref-512-2-msle-g-relu/decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2-
+dec-ref-512-2-msle-g-relu/decoder_d1/MatMulü
;dec-ref-512-2-msle-g-relu/decoder_d1/BiasAdd/ReadVariableOpReadVariableOpDdec_ref_512_2_msle_g_relu_decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;dec-ref-512-2-msle-g-relu/decoder_d1/BiasAdd/ReadVariableOp
,dec-ref-512-2-msle-g-relu/decoder_d1/BiasAddBiasAdd5dec-ref-512-2-msle-g-relu/decoder_d1/MatMul:product:0Cdec-ref-512-2-msle-g-relu/decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2.
,dec-ref-512-2-msle-g-relu/decoder_d1/BiasAddĢ
+dec-ref-512-2-msle-g-relu/activation_1/ReluRelu5dec-ref-512-2-msle-g-relu/decoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2-
+dec-ref-512-2-msle-g-relu/activation_1/Relu
>dec-ref-512-2-msle-g-relu/decoder_output/MatMul/ReadVariableOpReadVariableOpGdec_ref_512_2_msle_g_relu_decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02@
>dec-ref-512-2-msle-g-relu/decoder_output/MatMul/ReadVariableOp¢
/dec-ref-512-2-msle-g-relu/decoder_output/MatMulMatMul9dec-ref-512-2-msle-g-relu/activation_1/Relu:activations:0Fdec-ref-512-2-msle-g-relu/decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’21
/dec-ref-512-2-msle-g-relu/decoder_output/MatMul
?dec-ref-512-2-msle-g-relu/decoder_output/BiasAdd/ReadVariableOpReadVariableOpHdec_ref_512_2_msle_g_relu_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02A
?dec-ref-512-2-msle-g-relu/decoder_output/BiasAdd/ReadVariableOp¦
0dec-ref-512-2-msle-g-relu/decoder_output/BiasAddBiasAdd9dec-ref-512-2-msle-g-relu/decoder_output/MatMul:product:0Gdec-ref-512-2-msle-g-relu/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’22
0dec-ref-512-2-msle-g-relu/decoder_output/BiasAddŻ
0dec-ref-512-2-msle-g-relu/decoder_output/SigmoidSigmoid9dec-ref-512-2-msle-g-relu/decoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’22
0dec-ref-512-2-msle-g-relu/decoder_output/Sigmoid
!tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *æÖ32#
!tf_op_layer_Maximum_1/Maximum_1/yĆ
tf_op_layer_Maximum_1/Maximum_1Maximuminputs*tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2!
tf_op_layer_Maximum_1/Maximum_1
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *æÖ32
tf_op_layer_Maximum/Maximum/yå
tf_op_layer_Maximum/MaximumMaximum4dec-ref-512-2-msle-g-relu/decoder_output/Sigmoid:y:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Maximum/MaximumŹ
 ref_enc_d1/MatMul/ReadVariableOpReadVariableOpCenc_ref_512_2_msle_g_relu_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 ref_enc_d1/MatMul/ReadVariableOp
ref_enc_d1/MatMulMatMulinputs(ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
ref_enc_d1/MatMulČ
!ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpDenc_ref_512_2_msle_g_relu_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ref_enc_d1/BiasAdd/ReadVariableOp®
ref_enc_d1/BiasAddBiasAddref_enc_d1/MatMul:product:0)ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
ref_enc_d1/BiasAdd
tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_1/AddV2_1/yŅ
tf_op_layer_AddV2_1/AddV2_1AddV2#tf_op_layer_Maximum_1/Maximum_1:z:0&tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_1/AddV2_1{
tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2/AddV2/yĀ
tf_op_layer_AddV2/AddV2AddV2tf_op_layer_Maximum/Maximum:z:0"tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2/AddV2z
activation/ReluReluref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
activation/Relu
tf_op_layer_Log/LogLogtf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Log/Log
tf_op_layer_Log_1/Log_1Logtf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Log_1/Log_1Ņ
#ref_z_log_var/MatMul/ReadVariableOpReadVariableOpFenc_ref_512_2_msle_g_relu_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#ref_z_log_var/MatMul/ReadVariableOp“
ref_z_log_var/MatMulMatMulactivation/Relu:activations:0+ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_log_var/MatMulŠ
$ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpGenc_ref_512_2_msle_g_relu_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ref_z_log_var/BiasAdd/ReadVariableOp¹
ref_z_log_var/BiasAddBiasAddref_z_log_var/MatMul:product:0,ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_log_var/BiasAddÉ
 ref_z_mean/MatMul/ReadVariableOpReadVariableOpCenc_ref_512_2_msle_g_relu_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ref_z_mean/MatMul/ReadVariableOp«
ref_z_mean/MatMulMatMulactivation/Relu:activations:0(ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_mean/MatMulĒ
!ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpDenc_ref_512_2_msle_g_relu_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ref_z_mean/BiasAdd/ReadVariableOp­
ref_z_mean/BiasAddBiasAddref_z_mean/MatMul:product:0)ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
ref_z_mean/BiasAddļ
/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifferencetf_op_layer_Log/Log:y:0tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’21
/tf_op_layer_SquaredDifference/SquaredDifference
tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_2/AddV2_2/xĢ
tf_op_layer_AddV2_2/AddV2_2AddV2&tf_op_layer_AddV2_2/AddV2_2/x:output:0ref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_2/AddV2_2
tf_op_layer_Square/SquareSquareref_z_mean/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Square/Square
tf_op_layer_Exp/ExpExpref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Exp/Exp
'tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2)
'tf_op_layer_Mean/Mean/reduction_indicesŚ
tf_op_layer_Mean/MeanMean3tf_op_layer_SquaredDifference/SquaredDifference:z:00tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mean/Mean²
tf_op_layer_Sub/SubSubtf_op_layer_AddV2_2/AddV2_2:z:0tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sub/Subs
tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD2
tf_op_layer_Mul/Mul/y®
tf_op_layer_Mul/MulMultf_op_layer_Mean/Mean:output:0tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mul/Mul¬
tf_op_layer_Sub_1/Sub_1Subtf_op_layer_Sub/Sub:z:0tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sub_1/Sub_1
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2'
%tf_op_layer_Sum/Sum/reduction_indices»
tf_op_layer_Sum/SumSumtf_op_layer_Sub_1/Sub_1:z:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sum/Sum{
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   æ2
tf_op_layer_Mul_1/Mul_1/yø
tf_op_layer_Mul_1/Mul_1Multf_op_layer_Sum/Sum:output:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mul_1/Mul_1¶
tf_op_layer_AddV2_3/AddV2_3AddV2tf_op_layer_Mul/Mul:z:0tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_3/AddV2_3¤
+tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Mean_1/Mean_1/reduction_indicesÅ
tf_op_layer_Mean_1/Mean_1Meantf_op_layer_AddV2_3/AddV2_3:z:04tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mean_1/Mean_1
IdentityIdentity4dec-ref-512-2-msle-g-relu/decoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identityi

Identity_1Identity"tf_op_layer_Mean_1/Mean_1:output:0*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:’’’’’’’’’:::::::::::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ī

.__inference_decoder_output_layer_call_fn_16092

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallż
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_150002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ā
¬
9__inference_dec-ref-512-2-msle-g-relu_layer_call_fn_15930

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_150502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
å

*__inference_ref_enc_d1_layer_call_fn_15962

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallł
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_147252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ļ
ń
9__inference_enc-ref-512-2-msle-g-relu_layer_call_fn_14946
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2¢StatefulPartitionedCallź
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_149272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:’’’’’’’’’::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:’’’’’’’’’
'
_user_specified_nameencoder_input
Ė
Æ
9__inference_dec-ref-512-2-msle-g-relu_layer_call_fn_15089
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCall	latent_inunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_150782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:’’’’’’’’’::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:’’’’’’’’’
#
_user_specified_name	latent_in
Ń
­
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_14764

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
øQ
Ö
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15309
encoder_input#
enc_ref_512_2_msle_g_relu_15243#
enc_ref_512_2_msle_g_relu_15245#
enc_ref_512_2_msle_g_relu_15247#
enc_ref_512_2_msle_g_relu_15249#
enc_ref_512_2_msle_g_relu_15251#
enc_ref_512_2_msle_g_relu_15253#
dec_ref_512_2_msle_g_relu_15258#
dec_ref_512_2_msle_g_relu_15260#
dec_ref_512_2_msle_g_relu_15262#
dec_ref_512_2_msle_g_relu_15264
identity

identity_1¢1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall¢1enc-ref-512-2-msle-g-relu/StatefulPartitionedCall¢"ref_enc_d1/StatefulPartitionedCall¢%ref_z_log_var/StatefulPartitionedCall¢"ref_z_mean/StatefulPartitionedCall¤
1enc-ref-512-2-msle-g-relu/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_ref_512_2_msle_g_relu_15243enc_ref_512_2_msle_g_relu_15245enc_ref_512_2_msle_g_relu_15247enc_ref_512_2_msle_g_relu_15249enc_ref_512_2_msle_g_relu_15251enc_ref_512_2_msle_g_relu_15253*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_1492723
1enc-ref-512-2-msle-g-relu/StatefulPartitionedCallä
1dec-ref-512-2-msle-g-relu/StatefulPartitionedCallStatefulPartitionedCall:enc-ref-512-2-msle-g-relu/StatefulPartitionedCall:output:2dec_ref_512_2_msle_g_relu_15258dec_ref_512_2_msle_g_relu_15260dec_ref_512_2_msle_g_relu_15262dec_ref_512_2_msle_g_relu_15264*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_1507823
1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall
!tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *æÖ32#
!tf_op_layer_Maximum_1/Maximum_1/yŹ
tf_op_layer_Maximum_1/Maximum_1Maximumencoder_input*tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2!
tf_op_layer_Maximum_1/Maximum_1
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *æÖ32
tf_op_layer_Maximum/Maximum/yė
tf_op_layer_Maximum/MaximumMaximum:dec-ref-512-2-msle-g-relu/StatefulPartitionedCall:output:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Maximum/MaximumÄ
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputenc_ref_512_2_msle_g_relu_15243enc_ref_512_2_msle_g_relu_15245*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_147252$
"ref_enc_d1/StatefulPartitionedCall
tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_1/AddV2_1/yŅ
tf_op_layer_AddV2_1/AddV2_1AddV2#tf_op_layer_Maximum_1/Maximum_1:z:0&tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_1/AddV2_1{
tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2/AddV2/yĀ
tf_op_layer_AddV2/AddV2AddV2tf_op_layer_Maximum/Maximum:z:0"tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2/AddV2
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_147462
activation/PartitionedCall
tf_op_layer_Log/LogLogtf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Log/Log
tf_op_layer_Log_1/Log_1Logtf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’2
tf_op_layer_Log_1/Log_1ā
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0enc_ref_512_2_msle_g_relu_15251enc_ref_512_2_msle_g_relu_15253*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCallŁ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0enc_ref_512_2_msle_g_relu_15247enc_ref_512_2_msle_g_relu_15249*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
"ref_z_mean/StatefulPartitionedCallļ
/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifferencetf_op_layer_Log/Log:y:0tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:’’’’’’’’’21
/tf_op_layer_SquaredDifference/SquaredDifference
tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_2/AddV2_2/xÜ
tf_op_layer_AddV2_2/AddV2_2AddV2&tf_op_layer_AddV2_2/AddV2_2/x:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_2/AddV2_2®
tf_op_layer_Square/SquareSquare+ref_z_mean/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Square/Square¢
tf_op_layer_Exp/ExpExp.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Exp/Exp
'tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2)
'tf_op_layer_Mean/Mean/reduction_indicesŚ
tf_op_layer_Mean/MeanMean3tf_op_layer_SquaredDifference/SquaredDifference:z:00tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mean/Mean²
tf_op_layer_Sub/SubSubtf_op_layer_AddV2_2/AddV2_2:z:0tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sub/Subs
tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD2
tf_op_layer_Mul/Mul/y®
tf_op_layer_Mul/MulMultf_op_layer_Mean/Mean:output:0tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mul/Mul¬
tf_op_layer_Sub_1/Sub_1Subtf_op_layer_Sub/Sub:z:0tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sub_1/Sub_1
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’2'
%tf_op_layer_Sum/Sum/reduction_indices»
tf_op_layer_Sum/SumSumtf_op_layer_Sub_1/Sub_1:z:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Sum/Sum{
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   æ2
tf_op_layer_Mul_1/Mul_1/yø
tf_op_layer_Mul_1/Mul_1Multf_op_layer_Sum/Sum:output:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_Mul_1/Mul_1¶
tf_op_layer_AddV2_3/AddV2_3AddV2tf_op_layer_Mul/Mul:z:0tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:’’’’’’’’’2
tf_op_layer_AddV2_3/AddV2_3¤
+tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Mean_1/Mean_1/reduction_indicesÅ
tf_op_layer_Mean_1/Mean_1Meantf_op_layer_AddV2_3/AddV2_3:z:04tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mean_1/Mean_1ä
add_loss/PartitionedCallPartitionedCall"tf_op_layer_Mean_1/Mean_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_152282
add_loss/PartitionedCallé
IdentityIdentity:dec-ref-512-2-msle-g-relu/StatefulPartitionedCall:output:02^dec-ref-512-2-msle-g-relu/StatefulPartitionedCall2^enc-ref-512-2-msle-g-relu/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

IdentityĀ

Identity_1Identity!add_loss/PartitionedCall:output:12^dec-ref-512-2-msle-g-relu/StatefulPartitionedCall2^enc-ref-512-2-msle-g-relu/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:’’’’’’’’’::::::::::2f
1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall1dec-ref-512-2-msle-g-relu/StatefulPartitionedCall2f
1enc-ref-512-2-msle-g-relu/StatefulPartitionedCall1enc-ref-512-2-msle-g-relu/StatefulPartitionedCall2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:W S
(
_output_shapes
:’’’’’’’’’
'
_user_specified_nameencoder_input
Ļ
÷
#__inference_signature_wrapper_15535
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallĆ
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_147112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:’’’’’’’’’::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:’’’’’’’’’
'
_user_specified_nameencoder_input
¹
±
I__inference_decoder_output_layer_call_and_return_conditional_losses_15000

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:’’’’’’’’’2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Sigmoid`
IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

m
@__inference_ref_z_layer_call_and_return_conditional_losses_14822

inputs
inputs_1
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_normal/stddevÆ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:’’’’’’’’’*
dtype02$
"random_normal/RandomStandardNormal«
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:’’’’’’’’’2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:’’’’’’’’’2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:’’’’’’’’’2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ń
­
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_16001

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¤
o
C__inference_add_loss_layer_call_and_return_conditional_losses_16015

inputs
identity

identity_1I
IdentityIdentityinputs*
T0*
_output_shapes
: 2

IdentityM

Identity_1Identityinputs*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
	

9__inference_vae-ref-512-2-msle-g-relu_layer_call_fn_15741

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:’’’’’’’’’: *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_153812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:’’’’’’’’’::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs"ÄL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ź
serving_default¶
H
encoder_input7
serving_default_encoder_input:0’’’’’’’’’N
dec-ref-512-2-msle-g-relu1
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:²ć
ÅŅ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
	optimizer
loss
regularization_losses
	variables
trainable_variables
 	keras_api
!
signatures
É__call__
Ź_default_save_signature
+Ė&call_and_return_all_conditional_losses"Ķ
_tf_keras_networkėĢ{"class_name": "Functional", "name": "vae-ref-512-2-msle-g-relu", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "vae-ref-512-2-msle-g-relu", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "enc-ref-512-2-msle-g-relu", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}, "name": "ref_z", "inbound_nodes": [[["ref_z_mean", 0, 0, {}], ["ref_z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["ref_z_mean", 0, 0], ["ref_z_log_var", 0, 0], ["ref_z", 0, 0]]}, "name": "enc-ref-512-2-msle-g-relu", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "dec-ref-512-2-msle-g-relu", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "name": "dec-ref-512-2-msle-g-relu", "inbound_nodes": [[["enc-ref-512-2-msle-g-relu", 1, 2, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["dec-ref-512-2-msle-g-relu/decoder_output/Sigmoid", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}, "name": "tf_op_layer_Maximum", "inbound_nodes": [[["dec-ref-512-2-msle-g-relu", 1, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum_1", "op": "Maximum", "input": ["encoder_input", "Maximum_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}, "name": "tf_op_layer_Maximum_1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2", "op": "AddV2", "input": ["Maximum", "AddV2/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_AddV2", "inbound_nodes": [[["tf_op_layer_Maximum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_1", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_1", "op": "AddV2", "input": ["Maximum_1", "AddV2_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_AddV2_1", "inbound_nodes": [[["tf_op_layer_Maximum_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_2", "op": "AddV2", "input": ["AddV2_2/x", "ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 1.0}}, "name": "tf_op_layer_AddV2_2", "inbound_nodes": [[["ref_z_log_var", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Square", "trainable": true, "dtype": "float32", "node_def": {"name": "Square", "op": "Square", "input": ["ref_z_mean/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Square", "inbound_nodes": [[["ref_z_mean", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Log", "trainable": true, "dtype": "float32", "node_def": {"name": "Log", "op": "Log", "input": ["AddV2"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Log", "inbound_nodes": [[["tf_op_layer_AddV2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Log_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Log_1", "op": "Log", "input": ["AddV2_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Log_1", "inbound_nodes": [[["tf_op_layer_AddV2_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["AddV2_2", "Square"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub", "inbound_nodes": [[["tf_op_layer_AddV2_2", 0, 0, {}], ["tf_op_layer_Square", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Exp", "trainable": true, "dtype": "float32", "node_def": {"name": "Exp", "op": "Exp", "input": ["ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Exp", "inbound_nodes": [[["ref_z_log_var", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "SquaredDifference", "trainable": true, "dtype": "float32", "node_def": {"name": "SquaredDifference", "op": "SquaredDifference", "input": ["Log", "Log_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_SquaredDifference", "inbound_nodes": [[["tf_op_layer_Log", 0, 0, {}], ["tf_op_layer_Log_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub_1", "op": "Sub", "input": ["Sub", "Exp"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub_1", "inbound_nodes": [[["tf_op_layer_Sub", 0, 0, {}], ["tf_op_layer_Exp", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean", "op": "Mean", "input": ["SquaredDifference", "Mean/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Mean", "inbound_nodes": [[["tf_op_layer_SquaredDifference", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["Sub_1", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Sum", "inbound_nodes": [[["tf_op_layer_Sub_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["Mean", "Mul/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 784.0}}, "name": "tf_op_layer_Mul", "inbound_nodes": [[["tf_op_layer_Mean", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["Sum", "Mul_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": -0.5}}, "name": "tf_op_layer_Mul_1", "inbound_nodes": [[["tf_op_layer_Sum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_3", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_3", "op": "AddV2", "input": ["Mul", "Mul_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_3", "inbound_nodes": [[["tf_op_layer_Mul", 0, 0, {}], ["tf_op_layer_Mul_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean_1", "op": "Mean", "input": ["AddV2_3", "Mean_1/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Mean_1", "inbound_nodes": [[["tf_op_layer_AddV2_3", 0, 0, {}]]]}, {"class_name": "AddLoss", "config": {"name": "add_loss", "trainable": true, "dtype": "float32", "unconditional": false}, "name": "add_loss", "inbound_nodes": [[["tf_op_layer_Mean_1", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["dec-ref-512-2-msle-g-relu", 1, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "vae-ref-512-2-msle-g-relu", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "enc-ref-512-2-msle-g-relu", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}, "name": "ref_z", "inbound_nodes": [[["ref_z_mean", 0, 0, {}], ["ref_z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["ref_z_mean", 0, 0], ["ref_z_log_var", 0, 0], ["ref_z", 0, 0]]}, "name": "enc-ref-512-2-msle-g-relu", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "dec-ref-512-2-msle-g-relu", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "name": "dec-ref-512-2-msle-g-relu", "inbound_nodes": [[["enc-ref-512-2-msle-g-relu", 1, 2, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["dec-ref-512-2-msle-g-relu/decoder_output/Sigmoid", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}, "name": "tf_op_layer_Maximum", "inbound_nodes": [[["dec-ref-512-2-msle-g-relu", 1, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum_1", "op": "Maximum", "input": ["encoder_input", "Maximum_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}, "name": "tf_op_layer_Maximum_1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2", "op": "AddV2", "input": ["Maximum", "AddV2/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_AddV2", "inbound_nodes": [[["tf_op_layer_Maximum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_1", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_1", "op": "AddV2", "input": ["Maximum_1", "AddV2_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_AddV2_1", "inbound_nodes": [[["tf_op_layer_Maximum_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_2", "op": "AddV2", "input": ["AddV2_2/x", "ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 1.0}}, "name": "tf_op_layer_AddV2_2", "inbound_nodes": [[["ref_z_log_var", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Square", "trainable": true, "dtype": "float32", "node_def": {"name": "Square", "op": "Square", "input": ["ref_z_mean/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Square", "inbound_nodes": [[["ref_z_mean", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Log", "trainable": true, "dtype": "float32", "node_def": {"name": "Log", "op": "Log", "input": ["AddV2"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Log", "inbound_nodes": [[["tf_op_layer_AddV2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Log_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Log_1", "op": "Log", "input": ["AddV2_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Log_1", "inbound_nodes": [[["tf_op_layer_AddV2_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["AddV2_2", "Square"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub", "inbound_nodes": [[["tf_op_layer_AddV2_2", 0, 0, {}], ["tf_op_layer_Square", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Exp", "trainable": true, "dtype": "float32", "node_def": {"name": "Exp", "op": "Exp", "input": ["ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Exp", "inbound_nodes": [[["ref_z_log_var", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "SquaredDifference", "trainable": true, "dtype": "float32", "node_def": {"name": "SquaredDifference", "op": "SquaredDifference", "input": ["Log", "Log_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_SquaredDifference", "inbound_nodes": [[["tf_op_layer_Log", 0, 0, {}], ["tf_op_layer_Log_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub_1", "op": "Sub", "input": ["Sub", "Exp"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub_1", "inbound_nodes": [[["tf_op_layer_Sub", 0, 0, {}], ["tf_op_layer_Exp", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean", "op": "Mean", "input": ["SquaredDifference", "Mean/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Mean", "inbound_nodes": [[["tf_op_layer_SquaredDifference", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["Sub_1", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Sum", "inbound_nodes": [[["tf_op_layer_Sub_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["Mean", "Mul/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 784.0}}, "name": "tf_op_layer_Mul", "inbound_nodes": [[["tf_op_layer_Mean", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["Sum", "Mul_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": -0.5}}, "name": "tf_op_layer_Mul_1", "inbound_nodes": [[["tf_op_layer_Sum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_3", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_3", "op": "AddV2", "input": ["Mul", "Mul_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_3", "inbound_nodes": [[["tf_op_layer_Mul", 0, 0, {}], ["tf_op_layer_Mul_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean_1", "op": "Mean", "input": ["AddV2_3", "Mean_1/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Mean_1", "inbound_nodes": [[["tf_op_layer_AddV2_3", 0, 0, {}]]]}, {"class_name": "AddLoss", "config": {"name": "add_loss", "trainable": true, "dtype": "float32", "unconditional": false}, "name": "add_loss", "inbound_nodes": [[["tf_op_layer_Mean_1", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["dec-ref-512-2-msle-g-relu", 1, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ł"ö
_tf_keras_input_layerÖ{"class_name": "InputLayer", "name": "encoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}}
ą)
layer-0
layer_with_weights-0
layer-1
layer-2
	layer_with_weights-1
	layer-3
layer_with_weights-2
layer-4
"layer-5
#regularization_losses
$	variables
%trainable_variables
&	keras_api
Ģ__call__
+Ķ&call_and_return_all_conditional_losses"³'
_tf_keras_network'{"class_name": "Functional", "name": "enc-ref-512-2-msle-g-relu", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "enc-ref-512-2-msle-g-relu", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}, "name": "ref_z", "inbound_nodes": [[["ref_z_mean", 0, 0, {}], ["ref_z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["ref_z_mean", 0, 0], ["ref_z_log_var", 0, 0], ["ref_z", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "enc-ref-512-2-msle-g-relu", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}, "name": "ref_z", "inbound_nodes": [[["ref_z_mean", 0, 0, {}], ["ref_z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["ref_z_mean", 0, 0], ["ref_z_log_var", 0, 0], ["ref_z", 0, 0]]}}}
ō
'layer-0
(layer_with_weights-0
(layer-1
)layer-2
*layer_with_weights-1
*layer-3
+regularization_losses
,	variables
-trainable_variables
.	keras_api
Ī__call__
+Ļ&call_and_return_all_conditional_losses"ū
_tf_keras_networkß{"class_name": "Functional", "name": "dec-ref-512-2-msle-g-relu", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "dec-ref-512-2-msle-g-relu", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "dec-ref-512-2-msle-g-relu", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}}}
ż

/kernel
0bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
Š__call__
+Ń&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dense", "name": "ref_enc_d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
Ó
5regularization_losses
6	variables
7trainable_variables
8	keras_api
Ņ__call__
+Ó&call_and_return_all_conditional_losses"Ā
_tf_keras_layerØ{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}

9	keras_api"
_tf_keras_layerķ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Maximum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["dec-ref-512-2-msle-g-relu/decoder_output/Sigmoid", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}}
ž
:	keras_api"ģ
_tf_keras_layerŅ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Maximum_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Maximum_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum_1", "op": "Maximum", "input": ["encoder_input", "Maximum_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}}


;kernel
<bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
Ō__call__
+Õ&call_and_return_all_conditional_losses"Ś
_tf_keras_layerĄ{"class_name": "Dense", "name": "ref_z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
ū

Akernel
Bbias
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
Ö__call__
+×&call_and_return_all_conditional_losses"Ō
_tf_keras_layerŗ{"class_name": "Dense", "name": "ref_z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
Ó
G	keras_api"Į
_tf_keras_layer§{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "AddV2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2", "op": "AddV2", "input": ["Maximum", "AddV2/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}}
Ż
H	keras_api"Ė
_tf_keras_layer±{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "AddV2_1", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_1", "op": "AddV2", "input": ["Maximum_1", "AddV2_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}}
é
I	keras_api"×
_tf_keras_layer½{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "AddV2_2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_2", "op": "AddV2", "input": ["AddV2_2/x", "ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 1.0}}}
Ļ
J	keras_api"½
_tf_keras_layer£{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Square", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Square", "trainable": true, "dtype": "float32", "node_def": {"name": "Square", "op": "Square", "input": ["ref_z_mean/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
¶
K	keras_api"¤
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Log", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Log", "trainable": true, "dtype": "float32", "node_def": {"name": "Log", "op": "Log", "input": ["AddV2"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
¾
L	keras_api"¬
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Log_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Log_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Log_1", "op": "Log", "input": ["AddV2_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
Ā
M	keras_api"°
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sub", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["AddV2_2", "Square"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
Ę
N	keras_api"“
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Exp", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Exp", "trainable": true, "dtype": "float32", "node_def": {"name": "Exp", "op": "Exp", "input": ["ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
õ
O	keras_api"ć
_tf_keras_layerÉ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_SquaredDifference", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "SquaredDifference", "trainable": true, "dtype": "float32", "node_def": {"name": "SquaredDifference", "op": "SquaredDifference", "input": ["Log", "Log_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
Į
P	keras_api"Æ
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sub_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Sub_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub_1", "op": "Sub", "input": ["Sub", "Exp"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
 
Q	keras_api"
_tf_keras_layerō{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Mean", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean", "op": "Mean", "input": ["SquaredDifference", "Mean/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}}

R	keras_api"ż
_tf_keras_layerć{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["Sub_1", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}}
Č
S	keras_api"¶
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Mul", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["Mean", "Mul/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 784.0}}}
Ī
T	keras_api"¼
_tf_keras_layer¢{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["Sum", "Mul_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": -0.5}}}
Ė
U	keras_api"¹
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "AddV2_3", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_3", "op": "AddV2", "input": ["Mul", "Mul_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}

V	keras_api"
_tf_keras_layeró{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mean_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Mean_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean_1", "op": "Mean", "input": ["AddV2_3", "Mean_1/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0]}}}
Ī
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
Ų__call__
+Ł&call_and_return_all_conditional_losses"½
_tf_keras_layer£{"class_name": "AddLoss", "name": "add_loss", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_loss", "trainable": true, "dtype": "float32", "unconditional": false}}

[iter

\beta_1

]beta_2
	^decay
_learning_rate/mµ0m¶;m·<møAm¹Bmŗ`m»am¼bm½cm¾/væ0vĄ;vĮ<vĀAvĆBvÄ`vÅavĘbvĒcvČ"
	optimizer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
f
/0
01
A2
B3
;4
<5
`6
a7
b8
c9"
trackable_list_wrapper
f
/0
01
A2
B3
;4
<5
`6
a7
b8
c9"
trackable_list_wrapper
Ī
dnon_trainable_variables
elayer_regularization_losses
regularization_losses
flayer_metrics
gmetrics
	variables
trainable_variables

hlayers
É__call__
Ź_default_save_signature
+Ė&call_and_return_all_conditional_losses
'Ė"call_and_return_conditional_losses"
_generic_user_object
-
Śserving_default"
signature_map
±
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
Ū__call__
+Ü&call_and_return_all_conditional_losses" 
_tf_keras_layer{"class_name": "Sampling", "name": "ref_z", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}}
 "
trackable_list_wrapper
J
/0
01
A2
B3
;4
<5"
trackable_list_wrapper
J
/0
01
A2
B3
;4
<5"
trackable_list_wrapper
°
mnon_trainable_variables
nlayer_regularization_losses
#regularization_losses
olayer_metrics
pmetrics
$	variables
%trainable_variables

qlayers
Ģ__call__
+Ķ&call_and_return_all_conditional_losses
'Ķ"call_and_return_conditional_losses"
_generic_user_object
ķ"ź
_tf_keras_input_layerŹ{"class_name": "InputLayer", "name": "latent_in", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}}
ł

`kernel
abias
rregularization_losses
s	variables
ttrainable_variables
u	keras_api
Ż__call__
+Ž&call_and_return_all_conditional_losses"Ņ
_tf_keras_layerø{"class_name": "Dense", "name": "decoder_d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
×
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
ß__call__
+ą&call_and_return_all_conditional_losses"Ę
_tf_keras_layer¬{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}


bkernel
cbias
zregularization_losses
{	variables
|trainable_variables
}	keras_api
į__call__
+ā&call_and_return_all_conditional_losses"ß
_tf_keras_layerÅ{"class_name": "Dense", "name": "decoder_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
 "
trackable_list_wrapper
<
`0
a1
b2
c3"
trackable_list_wrapper
<
`0
a1
b2
c3"
trackable_list_wrapper
³
~non_trainable_variables
layer_regularization_losses
+regularization_losses
layer_metrics
metrics
,	variables
-trainable_variables
layers
Ī__call__
+Ļ&call_and_return_all_conditional_losses
'Ļ"call_and_return_conditional_losses"
_generic_user_object
%:#
2ref_enc_d1/kernel
:2ref_enc_d1/bias
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
µ
non_trainable_variables
 layer_regularization_losses
1regularization_losses
layer_metrics
metrics
2	variables
3trainable_variables
layers
Š__call__
+Ń&call_and_return_all_conditional_losses
'Ń"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
 layer_regularization_losses
5regularization_losses
layer_metrics
metrics
6	variables
7trainable_variables
layers
Ņ__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
':%	2ref_z_log_var/kernel
 :2ref_z_log_var/bias
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
µ
non_trainable_variables
 layer_regularization_losses
=regularization_losses
layer_metrics
metrics
>	variables
?trainable_variables
layers
Ō__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
$:"	2ref_z_mean/kernel
:2ref_z_mean/bias
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
µ
non_trainable_variables
 layer_regularization_losses
Cregularization_losses
layer_metrics
metrics
D	variables
Etrainable_variables
layers
Ö__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
 layer_regularization_losses
Wregularization_losses
layer_metrics
metrics
X	variables
Ytrainable_variables
layers
Ų__call__
+Ł&call_and_return_all_conditional_losses
'Ł"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
$:"	2decoder_d1/kernel
:2decoder_d1/bias
):'
2decoder_output/kernel
": 2decoder_output/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
0"
trackable_list_wrapper
ę
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
 layer_regularization_losses
iregularization_losses
layer_metrics
 metrics
j	variables
ktrainable_variables
”layers
Ū__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
0
1
2
	3
4
"5"
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
µ
¢non_trainable_variables
 £layer_regularization_losses
rregularization_losses
¤layer_metrics
„metrics
s	variables
ttrainable_variables
¦layers
Ż__call__
+Ž&call_and_return_all_conditional_losses
'Ž"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
§non_trainable_variables
 Ølayer_regularization_losses
vregularization_losses
©layer_metrics
Ŗmetrics
w	variables
xtrainable_variables
«layers
ß__call__
+ą&call_and_return_all_conditional_losses
'ą"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
µ
¬non_trainable_variables
 ­layer_regularization_losses
zregularization_losses
®layer_metrics
Æmetrics
{	variables
|trainable_variables
°layers
į__call__
+ā&call_and_return_all_conditional_losses
'ā"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
'0
(1
)2
*3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
æ

±total

²count
³	variables
“	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
±0
²1"
trackable_list_wrapper
.
³	variables"
_generic_user_object
*:(
2Adam/ref_enc_d1/kernel/m
#:!2Adam/ref_enc_d1/bias/m
,:*	2Adam/ref_z_log_var/kernel/m
%:#2Adam/ref_z_log_var/bias/m
):'	2Adam/ref_z_mean/kernel/m
": 2Adam/ref_z_mean/bias/m
):'	2Adam/decoder_d1/kernel/m
#:!2Adam/decoder_d1/bias/m
.:,
2Adam/decoder_output/kernel/m
':%2Adam/decoder_output/bias/m
*:(
2Adam/ref_enc_d1/kernel/v
#:!2Adam/ref_enc_d1/bias/v
,:*	2Adam/ref_z_log_var/kernel/v
%:#2Adam/ref_z_log_var/bias/v
):'	2Adam/ref_z_mean/kernel/v
": 2Adam/ref_z_mean/bias/v
):'	2Adam/decoder_d1/kernel/v
#:!2Adam/decoder_d1/bias/v
.:,
2Adam/decoder_output/kernel/v
':%2Adam/decoder_output/bias/v
²2Æ
9__inference_vae-ref-512-2-msle-g-relu_layer_call_fn_15767
9__inference_vae-ref-512-2-msle-g-relu_layer_call_fn_15741
9__inference_vae-ref-512-2-msle-g-relu_layer_call_fn_15500
9__inference_vae-ref-512-2-msle-g-relu_layer_call_fn_15405Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
å2ā
 __inference__wrapped_model_14711½
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *-¢*
(%
encoder_input’’’’’’’’’
2
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15309
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15625
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15715
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15240Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
²2Æ
9__inference_enc-ref-512-2-msle-g-relu_layer_call_fn_15881
9__inference_enc-ref-512-2-msle-g-relu_layer_call_fn_15860
9__inference_enc-ref-512-2-msle-g-relu_layer_call_fn_14946
9__inference_enc-ref-512-2-msle-g-relu_layer_call_fn_14902Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15839
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_14834
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15803
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_14857Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
²2Æ
9__inference_dec-ref-512-2-msle-g-relu_layer_call_fn_15930
9__inference_dec-ref-512-2-msle-g-relu_layer_call_fn_15061
9__inference_dec-ref-512-2-msle-g-relu_layer_call_fn_15943
9__inference_dec-ref-512-2-msle-g-relu_layer_call_fn_15089Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
2
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15017
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15899
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15032
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15917Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
Ō2Ń
*__inference_ref_enc_d1_layer_call_fn_15962¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ļ2ģ
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_15953¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ō2Ń
*__inference_activation_layer_call_fn_15972¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ļ2ģ
E__inference_activation_layer_call_and_return_conditional_losses_15967¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
×2Ō
-__inference_ref_z_log_var_layer_call_fn_15991¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ņ2ļ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_15982¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ō2Ń
*__inference_ref_z_mean_layer_call_fn_16010¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ļ2ģ
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_16001¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ņ2Ļ
(__inference_add_loss_layer_call_fn_16021¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ķ2ź
C__inference_add_loss_layer_call_and_return_conditional_losses_16015¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
8B6
#__inference_signature_wrapper_15535encoder_input
Ļ2Ģ
%__inference_ref_z_layer_call_fn_16043¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ź2ē
@__inference_ref_z_layer_call_and_return_conditional_losses_16037¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ō2Ń
*__inference_decoder_d1_layer_call_fn_16062¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ļ2ģ
E__inference_decoder_d1_layer_call_and_return_conditional_losses_16053¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ö2Ó
,__inference_activation_1_layer_call_fn_16072¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ń2ī
G__inference_activation_1_layer_call_and_return_conditional_losses_16067¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ų2Õ
.__inference_decoder_output_layer_call_fn_16092¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ó2š
I__inference_decoder_output_layer_call_and_return_conditional_losses_16083¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 Ā
 __inference__wrapped_model_14711
/0AB;<`abc7¢4
-¢*
(%
encoder_input’’’’’’’’’
Ŗ "VŖS
Q
dec-ref-512-2-msle-g-relu41
dec-ref-512-2-msle-g-relu’’’’’’’’’„
G__inference_activation_1_layer_call_and_return_conditional_losses_16067Z0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 }
,__inference_activation_1_layer_call_fn_16072M0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’£
E__inference_activation_layer_call_and_return_conditional_losses_15967Z0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 {
*__inference_activation_layer_call_fn_15972M0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
C__inference_add_loss_layer_call_and_return_conditional_losses_16015D¢
¢

inputs 
Ŗ ""¢


0 

	
1/0 U
(__inference_add_loss_layer_call_fn_16021)¢
¢

inputs 
Ŗ " Ā
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15017j`abc:¢7
0¢-
# 
	latent_in’’’’’’’’’
p

 
Ŗ "&¢#

0’’’’’’’’’
 Ā
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15032j`abc:¢7
0¢-
# 
	latent_in’’’’’’’’’
p 

 
Ŗ "&¢#

0’’’’’’’’’
 æ
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15899g`abc7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "&¢#

0’’’’’’’’’
 æ
T__inference_dec-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15917g`abc7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "&¢#

0’’’’’’’’’
 
9__inference_dec-ref-512-2-msle-g-relu_layer_call_fn_15061]`abc:¢7
0¢-
# 
	latent_in’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
9__inference_dec-ref-512-2-msle-g-relu_layer_call_fn_15089]`abc:¢7
0¢-
# 
	latent_in’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
9__inference_dec-ref-512-2-msle-g-relu_layer_call_fn_15930Z`abc7¢4
-¢*
 
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
9__inference_dec-ref-512-2-msle-g-relu_layer_call_fn_15943Z`abc7¢4
-¢*
 
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’¦
E__inference_decoder_d1_layer_call_and_return_conditional_losses_16053]`a/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 ~
*__inference_decoder_d1_layer_call_fn_16062P`a/¢,
%¢"
 
inputs’’’’’’’’’
Ŗ "’’’’’’’’’«
I__inference_decoder_output_layer_call_and_return_conditional_losses_16083^bc0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 
.__inference_decoder_output_layer_call_fn_16092Qbc0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_14834µ/0AB;<?¢<
5¢2
(%
encoder_input’’’’’’’’’
p

 
Ŗ "j¢g
`]

0/0’’’’’’’’’

0/1’’’’’’’’’

0/2’’’’’’’’’
 
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_14857µ/0AB;<?¢<
5¢2
(%
encoder_input’’’’’’’’’
p 

 
Ŗ "j¢g
`]

0/0’’’’’’’’’

0/1’’’’’’’’’

0/2’’’’’’’’’
 
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15803®/0AB;<8¢5
.¢+
!
inputs’’’’’’’’’
p

 
Ŗ "j¢g
`]

0/0’’’’’’’’’

0/1’’’’’’’’’

0/2’’’’’’’’’
 
T__inference_enc-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15839®/0AB;<8¢5
.¢+
!
inputs’’’’’’’’’
p 

 
Ŗ "j¢g
`]

0/0’’’’’’’’’

0/1’’’’’’’’’

0/2’’’’’’’’’
 ć
9__inference_enc-ref-512-2-msle-g-relu_layer_call_fn_14902„/0AB;<?¢<
5¢2
(%
encoder_input’’’’’’’’’
p

 
Ŗ "ZW

0’’’’’’’’’

1’’’’’’’’’

2’’’’’’’’’ć
9__inference_enc-ref-512-2-msle-g-relu_layer_call_fn_14946„/0AB;<?¢<
5¢2
(%
encoder_input’’’’’’’’’
p 

 
Ŗ "ZW

0’’’’’’’’’

1’’’’’’’’’

2’’’’’’’’’Ü
9__inference_enc-ref-512-2-msle-g-relu_layer_call_fn_15860/0AB;<8¢5
.¢+
!
inputs’’’’’’’’’
p

 
Ŗ "ZW

0’’’’’’’’’

1’’’’’’’’’

2’’’’’’’’’Ü
9__inference_enc-ref-512-2-msle-g-relu_layer_call_fn_15881/0AB;<8¢5
.¢+
!
inputs’’’’’’’’’
p 

 
Ŗ "ZW

0’’’’’’’’’

1’’’’’’’’’

2’’’’’’’’’§
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_15953^/00¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 
*__inference_ref_enc_d1_layer_call_fn_15962Q/00¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’Č
@__inference_ref_z_layer_call_and_return_conditional_losses_16037Z¢W
P¢M
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 
%__inference_ref_z_layer_call_fn_16043vZ¢W
P¢M
KH
"
inputs/0’’’’’’’’’
"
inputs/1’’’’’’’’’
Ŗ "’’’’’’’’’©
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_15982];<0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 
-__inference_ref_z_log_var_layer_call_fn_15991P;<0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’¦
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_16001]AB0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’
 ~
*__inference_ref_z_mean_layer_call_fn_16010PAB0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’Ö
#__inference_signature_wrapper_15535®
/0AB;<`abcH¢E
¢ 
>Ŗ;
9
encoder_input(%
encoder_input’’’’’’’’’"VŖS
Q
dec-ref-512-2-msle-g-relu41
dec-ref-512-2-msle-g-relu’’’’’’’’’Ü
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15240
/0AB;<`abc?¢<
5¢2
(%
encoder_input’’’’’’’’’
p

 
Ŗ "4¢1

0’’’’’’’’’

	
1/0 Ü
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15309
/0AB;<`abc?¢<
5¢2
(%
encoder_input’’’’’’’’’
p 

 
Ŗ "4¢1

0’’’’’’’’’

	
1/0 Ō
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15625|
/0AB;<`abc8¢5
.¢+
!
inputs’’’’’’’’’
p

 
Ŗ "4¢1

0’’’’’’’’’

	
1/0 Ō
T__inference_vae-ref-512-2-msle-g-relu_layer_call_and_return_conditional_losses_15715|
/0AB;<`abc8¢5
.¢+
!
inputs’’’’’’’’’
p 

 
Ŗ "4¢1

0’’’’’’’’’

	
1/0 „
9__inference_vae-ref-512-2-msle-g-relu_layer_call_fn_15405h
/0AB;<`abc?¢<
5¢2
(%
encoder_input’’’’’’’’’
p

 
Ŗ "’’’’’’’’’„
9__inference_vae-ref-512-2-msle-g-relu_layer_call_fn_15500h
/0AB;<`abc?¢<
5¢2
(%
encoder_input’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
9__inference_vae-ref-512-2-msle-g-relu_layer_call_fn_15741a
/0AB;<`abc8¢5
.¢+
!
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’
9__inference_vae-ref-512-2-msle-g-relu_layer_call_fn_15767a
/0AB;<`abc8¢5
.¢+
!
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’