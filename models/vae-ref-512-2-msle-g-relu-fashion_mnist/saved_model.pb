Тз
нЃ
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
О
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
 "serve*2.4.0-dev202007052v1.12.1-35711-g8202ae9d5c8ку
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
ЫK
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*K
valueќJBљJ BђJ
п
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
	variables
regularization_losses
trainable_variables
 	keras_api
!
signatures
 
ю
layer-0
layer_with_weights-0
layer-1
layer-2
	layer_with_weights-1
	layer-3
layer_with_weights-2
layer-4
"layer-5
#	variables
$regularization_losses
%trainable_variables
&	keras_api
К
'layer-0
(layer_with_weights-0
(layer-1
)layer-2
*layer_with_weights-1
*layer-3
+	variables
,regularization_losses
-trainable_variables
.	keras_api
h

/kernel
0bias
1	variables
2regularization_losses
3trainable_variables
4	keras_api
R
5	variables
6regularization_losses
7trainable_variables
8	keras_api

9	keras_api

:	keras_api
h

;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
h

Akernel
Bbias
C	variables
Dregularization_losses
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
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api

[iter

\beta_1

]beta_2
	^decay
_learning_rate/mЕ0mЖ;mЗ<mИAmЙBmК`mЛamМbmНcmО/vП0vР;vС<vТAvУBvФ`vХavЦbvЧcvШ
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
­
dlayer_regularization_losses
	variables
enon_trainable_variables
regularization_losses
fmetrics

glayers
trainable_variables
hlayer_metrics
 
R
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
*
/0
01
A2
B3
;4
<5
 
*
/0
01
A2
B3
;4
<5
­
mlayer_regularization_losses
#	variables
nnon_trainable_variables
$regularization_losses
ometrics

players
%trainable_variables
qlayer_metrics
 
h

`kernel
abias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
R
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
h

bkernel
cbias
z	variables
{regularization_losses
|trainable_variables
}	keras_api

`0
a1
b2
c3
 

`0
a1
b2
c3
А
~layer_regularization_losses
+	variables
non_trainable_variables
,regularization_losses
metrics
layers
-trainable_variables
layer_metrics
][
VARIABLE_VALUEref_enc_d1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEref_enc_d1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01
 

/0
01
В
 layer_regularization_losses
1	variables
non_trainable_variables
2regularization_losses
metrics
layers
3trainable_variables
layer_metrics
 
 
 
В
 layer_regularization_losses
5	variables
non_trainable_variables
6regularization_losses
metrics
layers
7trainable_variables
layer_metrics
 
 
`^
VARIABLE_VALUEref_z_log_var/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEref_z_log_var/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
 

;0
<1
В
 layer_regularization_losses
=	variables
non_trainable_variables
>regularization_losses
metrics
layers
?trainable_variables
layer_metrics
][
VARIABLE_VALUEref_z_mean/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEref_z_mean/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1
 

A0
B1
В
 layer_regularization_losses
C	variables
non_trainable_variables
Dregularization_losses
metrics
layers
Etrainable_variables
layer_metrics
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
В
 layer_regularization_losses
W	variables
non_trainable_variables
Xregularization_losses
metrics
layers
Ytrainable_variables
layer_metrics
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

0
Ц
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
 
В
 layer_regularization_losses
i	variables
non_trainable_variables
jregularization_losses
metrics
 layers
ktrainable_variables
Ёlayer_metrics
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
 

`0
a1
В
 Ђlayer_regularization_losses
r	variables
Ѓnon_trainable_variables
sregularization_losses
Єmetrics
Ѕlayers
ttrainable_variables
Іlayer_metrics
 
 
 
В
 Їlayer_regularization_losses
v	variables
Јnon_trainable_variables
wregularization_losses
Љmetrics
Њlayers
xtrainable_variables
Ћlayer_metrics

b0
c1
 

b0
c1
В
 Ќlayer_regularization_losses
z	variables
­non_trainable_variables
{regularization_losses
Ўmetrics
Џlayers
|trainable_variables
Аlayer_metrics
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
 
8

Бtotal

Вcount
Г	variables
Д	keras_api
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
Б0
В1

Г	variables
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
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_encoder_inputref_enc_d1/kernelref_enc_d1/biasref_z_mean/kernelref_z_mean/biasref_z_log_var/kernelref_z_log_var/biasdecoder_d1/kerneldecoder_d1/biasdecoder_output/kerneldecoder_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*,
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
#__inference_signature_wrapper_13484
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ы
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
__inference__traced_save_14175
т
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
!__inference__traced_restore_14296ЧА
Э
Џ
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13027

inputs
decoder_d1_13015
decoder_d1_13017
decoder_output_13021
decoder_output_13023
identityЂ"decoder_d1/StatefulPartitionedCallЂ&decoder_output/StatefulPartitionedCall
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_d1_13015decoder_d1_13017*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_129092$
"decoder_d1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_129302
activation_1/PartitionedCallв
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_13021decoder_output_13023*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_129492(
&decoder_output/StatefulPartitionedCallв
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
­
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_12674

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
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
P

__inference__traced_save_14175
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

identity_1ЂMergeV2Checkpoints
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
value3B1 B+_temp_6570fb9493634a6983964713f90815a1/part2	
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameр
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ђ
valueшBх&B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesд
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesы
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_ref_enc_d1_kernel_read_readvariableop*savev2_ref_enc_d1_bias_read_readvariableop/savev2_ref_z_log_var_kernel_read_readvariableop-savev2_ref_z_log_var_bias_read_readvariableop,savev2_ref_z_mean_kernel_read_readvariableop*savev2_ref_z_mean_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop,savev2_decoder_d1_kernel_read_readvariableop*savev2_decoder_d1_bias_read_readvariableop0savev2_decoder_output_kernel_read_readvariableop.savev2_decoder_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_ref_enc_d1_kernel_m_read_readvariableop1savev2_adam_ref_enc_d1_bias_m_read_readvariableop6savev2_adam_ref_z_log_var_kernel_m_read_readvariableop4savev2_adam_ref_z_log_var_bias_m_read_readvariableop3savev2_adam_ref_z_mean_kernel_m_read_readvariableop1savev2_adam_ref_z_mean_bias_m_read_readvariableop3savev2_adam_decoder_d1_kernel_m_read_readvariableop1savev2_adam_decoder_d1_bias_m_read_readvariableop7savev2_adam_decoder_output_kernel_m_read_readvariableop5savev2_adam_decoder_output_bias_m_read_readvariableop3savev2_adam_ref_enc_d1_kernel_v_read_readvariableop1savev2_adam_ref_enc_d1_bias_v_read_readvariableop6savev2_adam_ref_z_log_var_kernel_v_read_readvariableop4savev2_adam_ref_z_log_var_bias_v_read_readvariableop3savev2_adam_ref_z_mean_kernel_v_read_readvariableop1savev2_adam_ref_z_mean_bias_v_read_readvariableop3savev2_adam_decoder_d1_kernel_v_read_readvariableop1savev2_adam_decoder_d1_bias_v_read_readvariableop7savev2_adam_decoder_output_kernel_v_read_readvariableop5savev2_adam_decoder_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*Е
_input_shapesЃ
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
у

*__inference_decoder_d1_layer_call_fn_14011

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_129092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ы
џ
G__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_12895
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2ЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_128762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameencoder_input
З%
Ф
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13752

inputs-
)ref_enc_d1_matmul_readvariableop_resource.
*ref_enc_d1_biasadd_readvariableop_resource-
)ref_z_mean_matmul_readvariableop_resource.
*ref_z_mean_biasadd_readvariableop_resource0
,ref_z_log_var_matmul_readvariableop_resource1
-ref_z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2А
 ref_enc_d1/MatMul/ReadVariableOpReadVariableOp)ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 ref_enc_d1/MatMul/ReadVariableOp
ref_enc_d1/MatMulMatMulinputs(ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ref_enc_d1/MatMulЎ
!ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOp*ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ref_enc_d1/BiasAdd/ReadVariableOpЎ
ref_enc_d1/BiasAddBiasAddref_enc_d1/MatMul:product:0)ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ref_enc_d1/BiasAddz
activation/ReluReluref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
activation/ReluЏ
 ref_z_mean/MatMul/ReadVariableOpReadVariableOp)ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ref_z_mean/MatMul/ReadVariableOpЋ
ref_z_mean/MatMulMatMulactivation/Relu:activations:0(ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_mean/MatMul­
!ref_z_mean/BiasAdd/ReadVariableOpReadVariableOp*ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ref_z_mean/BiasAdd/ReadVariableOp­
ref_z_mean/BiasAddBiasAddref_z_mean/MatMul:product:0)ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_mean/BiasAddИ
#ref_z_log_var/MatMul/ReadVariableOpReadVariableOp,ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#ref_z_log_var/MatMul/ReadVariableOpД
ref_z_log_var/MatMulMatMulactivation/Relu:activations:0+ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_log_var/MatMulЖ
$ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOp-ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ref_z_log_var/BiasAdd/ReadVariableOpЙ
ref_z_log_var/BiasAddBiasAddref_z_log_var/MatMul:product:0,ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
ref_z/random_normal/stddevС
(ref_z/random_normal/RandomStandardNormalRandomStandardNormalref_z/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02*
(ref_z/random_normal/RandomStandardNormalУ
ref_z/random_normal/mulMul1ref_z/random_normal/RandomStandardNormal:output:0#ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z/random_normal/mulЃ
ref_z/random_normalAddref_z/random_normal/mul:z:0!ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
	ref_z/mul^
	ref_z/ExpExpref_z/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	ref_z/Exp{
ref_z/mul_1Mulref_z/Exp:y:0ref_z/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z/mul_1
	ref_z/addAddV2ref_z_mean/BiasAdd:output:0ref_z/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	ref_z/addo
IdentityIdentityref_z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityv

Identity_1Identityref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1e

Identity_2Identityref_z/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:џџџџџџџџџ:::::::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
ј
G__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13830

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2ЂStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_128762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
­
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_13902

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
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ю

.__inference_decoder_output_layer_call_fn_14041

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_129492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
б
­
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_12713

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
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я
ї
#__inference_signature_wrapper_13484
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
identityЂStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*,
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
 __inference__wrapped_model_126602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:џџџџџџџџџ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameencoder_input
ж
В
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12981
	latent_in
decoder_d1_12969
decoder_d1_12971
decoder_output_12975
decoder_output_12977
identityЂ"decoder_d1/StatefulPartitionedCallЂ&decoder_output/StatefulPartitionedCallЂ
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCall	latent_indecoder_d1_12969decoder_d1_12971*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_129092$
"decoder_d1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_129302
activation_1/PartitionedCallв
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_12975decoder_output_12977*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_129492(
&decoder_output/StatefulPartitionedCallв
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	latent_in
в
D
(__inference_add_loss_layer_call_fn_13970

inputs
identityЖ
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
C__inference_add_loss_layer_call_and_return_conditional_losses_131772
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
ОV

b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13189
encoder_input1
-enc_ref_512_2_msle_g_relu_fashion_mnist_130841
-enc_ref_512_2_msle_g_relu_fashion_mnist_130861
-enc_ref_512_2_msle_g_relu_fashion_mnist_130881
-enc_ref_512_2_msle_g_relu_fashion_mnist_130901
-enc_ref_512_2_msle_g_relu_fashion_mnist_130921
-enc_ref_512_2_msle_g_relu_fashion_mnist_130941
-dec_ref_512_2_msle_g_relu_fashion_mnist_131251
-dec_ref_512_2_msle_g_relu_fashion_mnist_131271
-dec_ref_512_2_msle_g_relu_fashion_mnist_131291
-dec_ref_512_2_msle_g_relu_fashion_mnist_13131
identity

identity_1Ђ?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ"ref_enc_d1/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCallЂ
?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallencoder_input-enc_ref_512_2_msle_g_relu_fashion_mnist_13084-enc_ref_512_2_msle_g_relu_fashion_mnist_13086-enc_ref_512_2_msle_g_relu_fashion_mnist_13088-enc_ref_512_2_msle_g_relu_fashion_mnist_13090-enc_ref_512_2_msle_g_relu_fashion_mnist_13092-enc_ref_512_2_msle_g_relu_fashion_mnist_13094*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_128322A
?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallд
?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallHenc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:2-dec_ref_512_2_msle_g_relu_fashion_mnist_13125-dec_ref_512_2_msle_g_relu_fashion_mnist_13127-dec_ref_512_2_msle_g_relu_fashion_mnist_13129-dec_ref_512_2_msle_g_relu_fashion_mnist_13131*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_129992A
?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall
!tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32#
!tf_op_layer_Maximum_1/Maximum_1/yЪ
tf_op_layer_Maximum_1/Maximum_1Maximumencoder_input*tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_Maximum_1/Maximum_1
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32
tf_op_layer_Maximum/Maximum/yљ
tf_op_layer_Maximum/MaximumMaximumHdec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Maximum/Maximumр
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallencoder_input-enc_ref_512_2_msle_g_relu_fashion_mnist_13084-enc_ref_512_2_msle_g_relu_fashion_mnist_13086*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_126742$
"ref_enc_d1/StatefulPartitionedCall
tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_1/AddV2_1/yв
tf_op_layer_AddV2_1/AddV2_1AddV2#tf_op_layer_Maximum_1/Maximum_1:z:0&tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_1/AddV2_1{
tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2/AddV2/yТ
tf_op_layer_AddV2/AddV2AddV2tf_op_layer_Maximum/Maximum:z:0"tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2/AddV2
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_126952
activation/PartitionedCall
tf_op_layer_Log/LogLogtf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Log/Log
tf_op_layer_Log_1/Log_1Logtf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Log_1/Log_1ў
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0-enc_ref_512_2_msle_g_relu_fashion_mnist_13092-enc_ref_512_2_msle_g_relu_fashion_mnist_13094*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_127392'
%ref_z_log_var/StatefulPartitionedCallѕ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0-enc_ref_512_2_msle_g_relu_fashion_mnist_13088-enc_ref_512_2_msle_g_relu_fashion_mnist_13090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_127132$
"ref_z_mean/StatefulPartitionedCallя
/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifferencetf_op_layer_Log/Log:y:0tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ21
/tf_op_layer_SquaredDifference/SquaredDifference
tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_2/AddV2_2/xм
tf_op_layer_AddV2_2/AddV2_2AddV2&tf_op_layer_AddV2_2/AddV2_2/x:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_2/AddV2_2Ў
tf_op_layer_Square/SquareSquare+ref_z_mean/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Square/SquareЂ
tf_op_layer_Exp/ExpExp.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Exp/Exp
'tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'tf_op_layer_Mean/Mean/reduction_indicesк
tf_op_layer_Mean/MeanMean3tf_op_layer_SquaredDifference/SquaredDifference:z:00tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mean/MeanВ
tf_op_layer_Sub/SubSubtf_op_layer_AddV2_2/AddV2_2:z:0tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sub/Subs
tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD2
tf_op_layer_Mul/Mul/yЎ
tf_op_layer_Mul/MulMultf_op_layer_Mean/Mean:output:0tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mul/MulЌ
tf_op_layer_Sub_1/Sub_1Subtf_op_layer_Sub/Sub:z:0tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sub_1/Sub_1
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%tf_op_layer_Sum/Sum/reduction_indicesЛ
tf_op_layer_Sum/SumSumtf_op_layer_Sub_1/Sub_1:z:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sum/Sum{
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   П2
tf_op_layer_Mul_1/Mul_1/yИ
tf_op_layer_Mul_1/Mul_1Multf_op_layer_Sum/Sum:output:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mul_1/Mul_1Ж
tf_op_layer_AddV2_3/AddV2_3AddV2tf_op_layer_Mul/Mul:z:0tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_3/AddV2_3Є
+tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Mean_1/Mean_1/reduction_indicesХ
tf_op_layer_Mean_1/Mean_1Meantf_op_layer_AddV2_3/AddV2_3:z:04tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mean_1/Mean_1ф
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
C__inference_add_loss_layer_call_and_return_conditional_losses_131772
add_loss/PartitionedCall
IdentityIdentityHdec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0@^dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall@^enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityо

Identity_1Identity!add_loss/PartitionedCall:output:1@^dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall@^enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:џџџџџџџџџ::::::::::2
?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall2
?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:W S
(
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameencoder_input
Й
c
G__inference_activation_1_layer_call_and_return_conditional_losses_12930

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
д
А
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_13931

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
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ы
џ
G__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_12851
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2ЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_128322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameencoder_input
ъ

-__inference_ref_z_log_var_layer_call_fn_13940

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_127392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
у

*__inference_ref_z_mean_layer_call_fn_13959

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_127132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є
o
C__inference_add_loss_layer_call_and_return_conditional_losses_13964

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
Й
Б
I__inference_decoder_output_layer_call_and_return_conditional_losses_12949

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
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Sigmoid`
IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
г
­
E__inference_decoder_d1_layer_call_and_return_conditional_losses_12909

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
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
г
­
E__inference_decoder_d1_layer_call_and_return_conditional_losses_14002

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
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ:::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
V

b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13425

inputs1
-enc_ref_512_2_msle_g_relu_fashion_mnist_133591
-enc_ref_512_2_msle_g_relu_fashion_mnist_133611
-enc_ref_512_2_msle_g_relu_fashion_mnist_133631
-enc_ref_512_2_msle_g_relu_fashion_mnist_133651
-enc_ref_512_2_msle_g_relu_fashion_mnist_133671
-enc_ref_512_2_msle_g_relu_fashion_mnist_133691
-dec_ref_512_2_msle_g_relu_fashion_mnist_133741
-dec_ref_512_2_msle_g_relu_fashion_mnist_133761
-dec_ref_512_2_msle_g_relu_fashion_mnist_133781
-dec_ref_512_2_msle_g_relu_fashion_mnist_13380
identity

identity_1Ђ?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ"ref_enc_d1/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCall
?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallinputs-enc_ref_512_2_msle_g_relu_fashion_mnist_13359-enc_ref_512_2_msle_g_relu_fashion_mnist_13361-enc_ref_512_2_msle_g_relu_fashion_mnist_13363-enc_ref_512_2_msle_g_relu_fashion_mnist_13365-enc_ref_512_2_msle_g_relu_fashion_mnist_13367-enc_ref_512_2_msle_g_relu_fashion_mnist_13369*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_128762A
?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallд
?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallHenc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:2-dec_ref_512_2_msle_g_relu_fashion_mnist_13374-dec_ref_512_2_msle_g_relu_fashion_mnist_13376-dec_ref_512_2_msle_g_relu_fashion_mnist_13378-dec_ref_512_2_msle_g_relu_fashion_mnist_13380*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_130272A
?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall
!tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32#
!tf_op_layer_Maximum_1/Maximum_1/yУ
tf_op_layer_Maximum_1/Maximum_1Maximuminputs*tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_Maximum_1/Maximum_1
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32
tf_op_layer_Maximum/Maximum/yљ
tf_op_layer_Maximum/MaximumMaximumHdec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Maximum/Maximumй
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallinputs-enc_ref_512_2_msle_g_relu_fashion_mnist_13359-enc_ref_512_2_msle_g_relu_fashion_mnist_13361*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_126742$
"ref_enc_d1/StatefulPartitionedCall
tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_1/AddV2_1/yв
tf_op_layer_AddV2_1/AddV2_1AddV2#tf_op_layer_Maximum_1/Maximum_1:z:0&tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_1/AddV2_1{
tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2/AddV2/yТ
tf_op_layer_AddV2/AddV2AddV2tf_op_layer_Maximum/Maximum:z:0"tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2/AddV2
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_126952
activation/PartitionedCall
tf_op_layer_Log/LogLogtf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Log/Log
tf_op_layer_Log_1/Log_1Logtf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Log_1/Log_1ў
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0-enc_ref_512_2_msle_g_relu_fashion_mnist_13367-enc_ref_512_2_msle_g_relu_fashion_mnist_13369*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_127392'
%ref_z_log_var/StatefulPartitionedCallѕ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0-enc_ref_512_2_msle_g_relu_fashion_mnist_13363-enc_ref_512_2_msle_g_relu_fashion_mnist_13365*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_127132$
"ref_z_mean/StatefulPartitionedCallя
/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifferencetf_op_layer_Log/Log:y:0tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ21
/tf_op_layer_SquaredDifference/SquaredDifference
tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_2/AddV2_2/xм
tf_op_layer_AddV2_2/AddV2_2AddV2&tf_op_layer_AddV2_2/AddV2_2/x:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_2/AddV2_2Ў
tf_op_layer_Square/SquareSquare+ref_z_mean/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Square/SquareЂ
tf_op_layer_Exp/ExpExp.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Exp/Exp
'tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'tf_op_layer_Mean/Mean/reduction_indicesк
tf_op_layer_Mean/MeanMean3tf_op_layer_SquaredDifference/SquaredDifference:z:00tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mean/MeanВ
tf_op_layer_Sub/SubSubtf_op_layer_AddV2_2/AddV2_2:z:0tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sub/Subs
tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD2
tf_op_layer_Mul/Mul/yЎ
tf_op_layer_Mul/MulMultf_op_layer_Mean/Mean:output:0tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mul/MulЌ
tf_op_layer_Sub_1/Sub_1Subtf_op_layer_Sub/Sub:z:0tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sub_1/Sub_1
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%tf_op_layer_Sum/Sum/reduction_indicesЛ
tf_op_layer_Sum/SumSumtf_op_layer_Sub_1/Sub_1:z:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sum/Sum{
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   П2
tf_op_layer_Mul_1/Mul_1/yИ
tf_op_layer_Mul_1/Mul_1Multf_op_layer_Sum/Sum:output:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mul_1/Mul_1Ж
tf_op_layer_AddV2_3/AddV2_3AddV2tf_op_layer_Mul/Mul:z:0tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_3/AddV2_3Є
+tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Mean_1/Mean_1/reduction_indicesХ
tf_op_layer_Mean_1/Mean_1Meantf_op_layer_AddV2_3/AddV2_3:z:04tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mean_1/Mean_1ф
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
C__inference_add_loss_layer_call_and_return_conditional_losses_131772
add_loss/PartitionedCall
IdentityIdentityHdec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0@^dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall@^enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityо

Identity_1Identity!add_loss/PartitionedCall:output:1@^dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall@^enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:џџџџџџџџџ::::::::::2
?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall2
?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ
Ф
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12806
encoder_input
ref_enc_d1_12786
ref_enc_d1_12788
ref_z_mean_12792
ref_z_mean_12794
ref_z_log_var_12797
ref_z_log_var_12799
identity

identity_1

identity_2Ђ"ref_enc_d1/StatefulPartitionedCallЂref_z/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCallІ
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputref_enc_d1_12786ref_enc_d1_12788*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_126742$
"ref_enc_d1/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_126952
activation/PartitionedCallЛ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_12792ref_z_mean_12794*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_127132$
"ref_z_mean/StatefulPartitionedCallЪ
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_12797ref_z_log_var_12799*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_127392'
%ref_z_log_var/StatefulPartitionedCallЛ
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_127712
ref_z/StatefulPartitionedCall
IdentityIdentity+ref_z_mean/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity&ref_z/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:џџџџџџџџџ::::::2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2>
ref_z/StatefulPartitionedCallref_z/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:W S
(
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameencoder_input
Ќ

b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13574

inputsU
Qenc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resourceV
Renc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resourceU
Qenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resourceV
Renc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resourceX
Tenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resourceY
Uenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resourceU
Qdec_ref_512_2_msle_g_relu_fashion_mnist_decoder_d1_matmul_readvariableop_resourceV
Rdec_ref_512_2_msle_g_relu_fashion_mnist_decoder_d1_biasadd_readvariableop_resourceY
Udec_ref_512_2_msle_g_relu_fashion_mnist_decoder_output_matmul_readvariableop_resourceZ
Vdec_ref_512_2_msle_g_relu_fashion_mnist_decoder_output_biasadd_readvariableop_resource
identity

identity_1Ј
Henc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOpReadVariableOpQenc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02J
Henc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp
9enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulMatMulinputsPenc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2;
9enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulІ
Ienc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpRenc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02K
Ienc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpЮ
:enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddBiasAddCenc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul:product:0Qenc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2<
:enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddђ
7enc-ref-512-2-msle-g-relu-fashion_mnist/activation/ReluReluCenc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ29
7enc-ref-512-2-msle-g-relu-fashion_mnist/activation/ReluЇ
Henc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpReadVariableOpQenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02J
Henc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpЫ
9enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMulMatMulEenc-ref-512-2-msle-g-relu-fashion_mnist/activation/Relu:activations:0Penc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2;
9enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMulЅ
Ienc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpRenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02K
Ienc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpЭ
:enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAddBiasAddCenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul:product:0Qenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2<
:enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAddА
Kenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpReadVariableOpTenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02M
Kenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpд
<enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulMatMulEenc-ref-512-2-msle-g-relu-fashion_mnist/activation/Relu:activations:0Senc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2>
<enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulЎ
Lenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpUenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02N
Lenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpй
=enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddBiasAddFenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul:product:0Tenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2?
=enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddн
3enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/ShapeShapeCenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd:output:0*
T0*
_output_shapes
:25
3enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/ShapeЩ
@enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2B
@enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/meanЭ
Benc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2D
Benc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/stddevЙ
Penc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormalRandomStandardNormal<enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02R
Penc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormalу
?enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/mulMulYenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormal:output:0Kenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2A
?enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/mulУ
;enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normalAddCenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/mul:z:0Ienc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2=
;enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normalЏ
3enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?25
3enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul/xЅ
1enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mulMul<enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul/x:output:0Fenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ23
1enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mulж
1enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/ExpExp5enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ23
1enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/Exp
3enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul_1Mul5enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/Exp:y:0?enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ25
3enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul_1
1enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/addAddV2Cenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd:output:07enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ23
1enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/addЇ
Hdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOpReadVariableOpQdec_ref_512_2_msle_g_relu_fashion_mnist_decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02J
Hdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOpМ
9dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMulMatMul5enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/add:z:0Pdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2;
9dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMulІ
Idec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOpReadVariableOpRdec_ref_512_2_msle_g_relu_fashion_mnist_decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02K
Idec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOpЮ
:dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAddBiasAddCdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMul:product:0Qdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2<
:dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAddі
9dec-ref-512-2-msle-g-relu-fashion_mnist/activation_1/ReluReluCdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2;
9dec-ref-512-2-msle-g-relu-fashion_mnist/activation_1/ReluД
Ldec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOpReadVariableOpUdec_ref_512_2_msle_g_relu_fashion_mnist_decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02N
Ldec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOpк
=dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMulMatMulGdec-ref-512-2-msle-g-relu-fashion_mnist/activation_1/Relu:activations:0Tdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2?
=dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMulВ
Mdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOpReadVariableOpVdec_ref_512_2_msle_g_relu_fashion_mnist_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02O
Mdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOpо
>dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAddBiasAddGdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMul:product:0Udec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2@
>dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAdd
>dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/SigmoidSigmoidGdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2@
>dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/Sigmoid
!tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32#
!tf_op_layer_Maximum_1/Maximum_1/yУ
tf_op_layer_Maximum_1/Maximum_1Maximuminputs*tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_Maximum_1/Maximum_1
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32
tf_op_layer_Maximum/Maximum/yѓ
tf_op_layer_Maximum/MaximumMaximumBdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/Sigmoid:y:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Maximum/Maximumи
 ref_enc_d1/MatMul/ReadVariableOpReadVariableOpQenc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 ref_enc_d1/MatMul/ReadVariableOp
ref_enc_d1/MatMulMatMulinputs(ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ref_enc_d1/MatMulж
!ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpRenc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ref_enc_d1/BiasAdd/ReadVariableOpЎ
ref_enc_d1/BiasAddBiasAddref_enc_d1/MatMul:product:0)ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ref_enc_d1/BiasAdd
tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_1/AddV2_1/yв
tf_op_layer_AddV2_1/AddV2_1AddV2#tf_op_layer_Maximum_1/Maximum_1:z:0&tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_1/AddV2_1{
tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2/AddV2/yТ
tf_op_layer_AddV2/AddV2AddV2tf_op_layer_Maximum/Maximum:z:0"tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2/AddV2z
activation/ReluReluref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
activation/Relu
tf_op_layer_Log/LogLogtf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Log/Log
tf_op_layer_Log_1/Log_1Logtf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Log_1/Log_1р
#ref_z_log_var/MatMul/ReadVariableOpReadVariableOpTenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#ref_z_log_var/MatMul/ReadVariableOpД
ref_z_log_var/MatMulMatMulactivation/Relu:activations:0+ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_log_var/MatMulо
$ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpUenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ref_z_log_var/BiasAdd/ReadVariableOpЙ
ref_z_log_var/BiasAddBiasAddref_z_log_var/MatMul:product:0,ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_log_var/BiasAddз
 ref_z_mean/MatMul/ReadVariableOpReadVariableOpQenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ref_z_mean/MatMul/ReadVariableOpЋ
ref_z_mean/MatMulMatMulactivation/Relu:activations:0(ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_mean/MatMulе
!ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpRenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ref_z_mean/BiasAdd/ReadVariableOp­
ref_z_mean/BiasAddBiasAddref_z_mean/MatMul:product:0)ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_mean/BiasAddя
/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifferencetf_op_layer_Log/Log:y:0tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ21
/tf_op_layer_SquaredDifference/SquaredDifference
tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_2/AddV2_2/xЬ
tf_op_layer_AddV2_2/AddV2_2AddV2&tf_op_layer_AddV2_2/AddV2_2/x:output:0ref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_2/AddV2_2
tf_op_layer_Square/SquareSquareref_z_mean/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Square/Square
tf_op_layer_Exp/ExpExpref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Exp/Exp
'tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'tf_op_layer_Mean/Mean/reduction_indicesк
tf_op_layer_Mean/MeanMean3tf_op_layer_SquaredDifference/SquaredDifference:z:00tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mean/MeanВ
tf_op_layer_Sub/SubSubtf_op_layer_AddV2_2/AddV2_2:z:0tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sub/Subs
tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD2
tf_op_layer_Mul/Mul/yЎ
tf_op_layer_Mul/MulMultf_op_layer_Mean/Mean:output:0tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mul/MulЌ
tf_op_layer_Sub_1/Sub_1Subtf_op_layer_Sub/Sub:z:0tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sub_1/Sub_1
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%tf_op_layer_Sum/Sum/reduction_indicesЛ
tf_op_layer_Sum/SumSumtf_op_layer_Sub_1/Sub_1:z:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sum/Sum{
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   П2
tf_op_layer_Mul_1/Mul_1/yИ
tf_op_layer_Mul_1/Mul_1Multf_op_layer_Sum/Sum:output:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mul_1/Mul_1Ж
tf_op_layer_AddV2_3/AddV2_3AddV2tf_op_layer_Mul/Mul:z:0tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_3/AddV2_3Є
+tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Mean_1/Mean_1/reduction_indicesХ
tf_op_layer_Mean_1/Mean_1Meantf_op_layer_AddV2_3/AddV2_3:z:04tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mean_1/Mean_1
IdentityIdentityBdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1Identity"tf_op_layer_Mean_1/Mean_1:output:0*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:џџџџџџџџџ:::::::::::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Й
Б
I__inference_decoder_output_layer_call_and_return_conditional_losses_14032

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
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Sigmoid`
IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ћ
!__inference__traced_restore_14296
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
identity_38ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9ц
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ђ
valueшBх&B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesк
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesь
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ў
_output_shapes
::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЁ
AssignVariableOpAssignVariableOp"assignvariableop_ref_enc_d1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ї
AssignVariableOp_1AssignVariableOp"assignvariableop_1_ref_enc_d1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ќ
AssignVariableOp_2AssignVariableOp'assignvariableop_2_ref_z_log_var_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Њ
AssignVariableOp_3AssignVariableOp%assignvariableop_3_ref_z_log_var_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Љ
AssignVariableOp_4AssignVariableOp$assignvariableop_4_ref_z_mean_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ї
AssignVariableOp_5AssignVariableOp"assignvariableop_5_ref_z_mean_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6Ё
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѓ
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ѓ
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ђ
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ў
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
Identity_12Ћ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_decoder_d1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Б
AssignVariableOp_13AssignVariableOp)assignvariableop_13_decoder_output_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Џ
AssignVariableOp_14AssignVariableOp'assignvariableop_14_decoder_output_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ё
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ё
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Д
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_ref_enc_d1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18В
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_ref_enc_d1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19З
AssignVariableOp_19AssignVariableOp/assignvariableop_19_adam_ref_z_log_var_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Е
AssignVariableOp_20AssignVariableOp-assignvariableop_20_adam_ref_z_log_var_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Д
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_ref_z_mean_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22В
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_ref_z_mean_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Д
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_decoder_d1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24В
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_decoder_d1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25И
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_decoder_output_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ж
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_decoder_output_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Д
AssignVariableOp_27AssignVariableOp,assignvariableop_27_adam_ref_enc_d1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28В
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_ref_enc_d1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29З
AssignVariableOp_29AssignVariableOp/assignvariableop_29_adam_ref_z_log_var_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Е
AssignVariableOp_30AssignVariableOp-assignvariableop_30_adam_ref_z_log_var_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Д
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_ref_z_mean_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32В
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_ref_z_mean_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Д
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_decoder_d1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34В
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_decoder_d1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35И
AssignVariableOp_35AssignVariableOp0assignvariableop_35_adam_decoder_output_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ж
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
Identity_37џ
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_38"#
identity_38Identity_38:output:0*Ћ
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
И	

G__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13354
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
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:џџџџџџџџџ: *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_133302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:џџџџџџџџџ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameencoder_input
Є
o
C__inference_add_loss_layer_call_and_return_conditional_losses_13177

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
д
А
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_12739

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
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

m
@__inference_ref_z_layer_call_and_return_conditional_losses_12771

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
random_normal/stddevЏ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02$
"random_normal/RandomStandardNormalЋ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ
Ф
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12783
encoder_input
ref_enc_d1_12685
ref_enc_d1_12687
ref_z_mean_12724
ref_z_mean_12726
ref_z_log_var_12750
ref_z_log_var_12752
identity

identity_1

identity_2Ђ"ref_enc_d1/StatefulPartitionedCallЂref_z/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCallІ
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputref_enc_d1_12685ref_enc_d1_12687*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_126742$
"ref_enc_d1/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_126952
activation/PartitionedCallЛ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_12724ref_z_mean_12726*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_127132$
"ref_z_mean/StatefulPartitionedCallЪ
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_12750ref_z_log_var_12752*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_127392'
%ref_z_log_var/StatefulPartitionedCallЛ
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_127712
ref_z/StatefulPartitionedCall
IdentityIdentity+ref_z_mean/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity&ref_z/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:џџџџџџџџџ::::::2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2>
ref_z/StatefulPartitionedCallref_z/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:W S
(
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameencoder_input
ч
Н
G__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13038
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCall	latent_inunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_130272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	latent_in
б
­
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_13950

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
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Э
Џ
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12999

inputs
decoder_d1_12987
decoder_d1_12989
decoder_output_12993
decoder_output_12995
identityЂ"decoder_d1/StatefulPartitionedCallЂ&decoder_output/StatefulPartitionedCall
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_d1_12987decoder_d1_12989*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_129092$
"decoder_d1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_129302
activation_1/PartitionedCallв
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_12993decoder_output_12995*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_129492(
&decoder_output/StatefulPartitionedCallв
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

F
*__inference_activation_layer_call_fn_13921

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_126952
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ОV

b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13258
encoder_input1
-enc_ref_512_2_msle_g_relu_fashion_mnist_131921
-enc_ref_512_2_msle_g_relu_fashion_mnist_131941
-enc_ref_512_2_msle_g_relu_fashion_mnist_131961
-enc_ref_512_2_msle_g_relu_fashion_mnist_131981
-enc_ref_512_2_msle_g_relu_fashion_mnist_132001
-enc_ref_512_2_msle_g_relu_fashion_mnist_132021
-dec_ref_512_2_msle_g_relu_fashion_mnist_132071
-dec_ref_512_2_msle_g_relu_fashion_mnist_132091
-dec_ref_512_2_msle_g_relu_fashion_mnist_132111
-dec_ref_512_2_msle_g_relu_fashion_mnist_13213
identity

identity_1Ђ?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ"ref_enc_d1/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCallЂ
?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallencoder_input-enc_ref_512_2_msle_g_relu_fashion_mnist_13192-enc_ref_512_2_msle_g_relu_fashion_mnist_13194-enc_ref_512_2_msle_g_relu_fashion_mnist_13196-enc_ref_512_2_msle_g_relu_fashion_mnist_13198-enc_ref_512_2_msle_g_relu_fashion_mnist_13200-enc_ref_512_2_msle_g_relu_fashion_mnist_13202*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_128762A
?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallд
?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallHenc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:2-dec_ref_512_2_msle_g_relu_fashion_mnist_13207-dec_ref_512_2_msle_g_relu_fashion_mnist_13209-dec_ref_512_2_msle_g_relu_fashion_mnist_13211-dec_ref_512_2_msle_g_relu_fashion_mnist_13213*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_130272A
?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall
!tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32#
!tf_op_layer_Maximum_1/Maximum_1/yЪ
tf_op_layer_Maximum_1/Maximum_1Maximumencoder_input*tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_Maximum_1/Maximum_1
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32
tf_op_layer_Maximum/Maximum/yљ
tf_op_layer_Maximum/MaximumMaximumHdec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Maximum/Maximumр
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallencoder_input-enc_ref_512_2_msle_g_relu_fashion_mnist_13192-enc_ref_512_2_msle_g_relu_fashion_mnist_13194*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_126742$
"ref_enc_d1/StatefulPartitionedCall
tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_1/AddV2_1/yв
tf_op_layer_AddV2_1/AddV2_1AddV2#tf_op_layer_Maximum_1/Maximum_1:z:0&tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_1/AddV2_1{
tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2/AddV2/yТ
tf_op_layer_AddV2/AddV2AddV2tf_op_layer_Maximum/Maximum:z:0"tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2/AddV2
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_126952
activation/PartitionedCall
tf_op_layer_Log/LogLogtf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Log/Log
tf_op_layer_Log_1/Log_1Logtf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Log_1/Log_1ў
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0-enc_ref_512_2_msle_g_relu_fashion_mnist_13200-enc_ref_512_2_msle_g_relu_fashion_mnist_13202*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_127392'
%ref_z_log_var/StatefulPartitionedCallѕ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0-enc_ref_512_2_msle_g_relu_fashion_mnist_13196-enc_ref_512_2_msle_g_relu_fashion_mnist_13198*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_127132$
"ref_z_mean/StatefulPartitionedCallя
/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifferencetf_op_layer_Log/Log:y:0tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ21
/tf_op_layer_SquaredDifference/SquaredDifference
tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_2/AddV2_2/xм
tf_op_layer_AddV2_2/AddV2_2AddV2&tf_op_layer_AddV2_2/AddV2_2/x:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_2/AddV2_2Ў
tf_op_layer_Square/SquareSquare+ref_z_mean/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Square/SquareЂ
tf_op_layer_Exp/ExpExp.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Exp/Exp
'tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'tf_op_layer_Mean/Mean/reduction_indicesк
tf_op_layer_Mean/MeanMean3tf_op_layer_SquaredDifference/SquaredDifference:z:00tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mean/MeanВ
tf_op_layer_Sub/SubSubtf_op_layer_AddV2_2/AddV2_2:z:0tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sub/Subs
tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD2
tf_op_layer_Mul/Mul/yЎ
tf_op_layer_Mul/MulMultf_op_layer_Mean/Mean:output:0tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mul/MulЌ
tf_op_layer_Sub_1/Sub_1Subtf_op_layer_Sub/Sub:z:0tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sub_1/Sub_1
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%tf_op_layer_Sum/Sum/reduction_indicesЛ
tf_op_layer_Sum/SumSumtf_op_layer_Sub_1/Sub_1:z:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sum/Sum{
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   П2
tf_op_layer_Mul_1/Mul_1/yИ
tf_op_layer_Mul_1/Mul_1Multf_op_layer_Sum/Sum:output:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mul_1/Mul_1Ж
tf_op_layer_AddV2_3/AddV2_3AddV2tf_op_layer_Mul/Mul:z:0tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_3/AddV2_3Є
+tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Mean_1/Mean_1/reduction_indicesХ
tf_op_layer_Mean_1/Mean_1Meantf_op_layer_AddV2_3/AddV2_3:z:04tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mean_1/Mean_1ф
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
C__inference_add_loss_layer_call_and_return_conditional_losses_131772
add_loss/PartitionedCall
IdentityIdentityHdec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0@^dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall@^enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityо

Identity_1Identity!add_loss/PartitionedCall:output:1@^dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall@^enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:џџџџџџџџџ::::::::::2
?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall2
?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:W S
(
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameencoder_input
Й
c
G__inference_activation_1_layer_call_and_return_conditional_losses_14016

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ќ

b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13664

inputsU
Qenc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resourceV
Renc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resourceU
Qenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resourceV
Renc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resourceX
Tenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resourceY
Uenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resourceU
Qdec_ref_512_2_msle_g_relu_fashion_mnist_decoder_d1_matmul_readvariableop_resourceV
Rdec_ref_512_2_msle_g_relu_fashion_mnist_decoder_d1_biasadd_readvariableop_resourceY
Udec_ref_512_2_msle_g_relu_fashion_mnist_decoder_output_matmul_readvariableop_resourceZ
Vdec_ref_512_2_msle_g_relu_fashion_mnist_decoder_output_biasadd_readvariableop_resource
identity

identity_1Ј
Henc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOpReadVariableOpQenc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02J
Henc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp
9enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulMatMulinputsPenc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2;
9enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulІ
Ienc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpRenc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02K
Ienc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpЮ
:enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddBiasAddCenc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul:product:0Qenc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2<
:enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddђ
7enc-ref-512-2-msle-g-relu-fashion_mnist/activation/ReluReluCenc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ29
7enc-ref-512-2-msle-g-relu-fashion_mnist/activation/ReluЇ
Henc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpReadVariableOpQenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02J
Henc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpЫ
9enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMulMatMulEenc-ref-512-2-msle-g-relu-fashion_mnist/activation/Relu:activations:0Penc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2;
9enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMulЅ
Ienc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpRenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02K
Ienc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpЭ
:enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAddBiasAddCenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul:product:0Qenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2<
:enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAddА
Kenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpReadVariableOpTenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02M
Kenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpд
<enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulMatMulEenc-ref-512-2-msle-g-relu-fashion_mnist/activation/Relu:activations:0Senc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2>
<enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulЎ
Lenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpUenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02N
Lenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpй
=enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddBiasAddFenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul:product:0Tenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2?
=enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddн
3enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/ShapeShapeCenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd:output:0*
T0*
_output_shapes
:25
3enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/ShapeЩ
@enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2B
@enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/meanЭ
Benc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2D
Benc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/stddevЙ
Penc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormalRandomStandardNormal<enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02R
Penc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormalу
?enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/mulMulYenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormal:output:0Kenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2A
?enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/mulУ
;enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normalAddCenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/mul:z:0Ienc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2=
;enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normalЏ
3enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?25
3enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul/xЅ
1enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mulMul<enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul/x:output:0Fenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ23
1enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mulж
1enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/ExpExp5enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ23
1enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/Exp
3enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul_1Mul5enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/Exp:y:0?enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ25
3enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul_1
1enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/addAddV2Cenc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd:output:07enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ23
1enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/addЇ
Hdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOpReadVariableOpQdec_ref_512_2_msle_g_relu_fashion_mnist_decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02J
Hdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOpМ
9dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMulMatMul5enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/add:z:0Pdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2;
9dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMulІ
Idec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOpReadVariableOpRdec_ref_512_2_msle_g_relu_fashion_mnist_decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02K
Idec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOpЮ
:dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAddBiasAddCdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMul:product:0Qdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2<
:dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAddі
9dec-ref-512-2-msle-g-relu-fashion_mnist/activation_1/ReluReluCdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2;
9dec-ref-512-2-msle-g-relu-fashion_mnist/activation_1/ReluД
Ldec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOpReadVariableOpUdec_ref_512_2_msle_g_relu_fashion_mnist_decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02N
Ldec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOpк
=dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMulMatMulGdec-ref-512-2-msle-g-relu-fashion_mnist/activation_1/Relu:activations:0Tdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2?
=dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMulВ
Mdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOpReadVariableOpVdec_ref_512_2_msle_g_relu_fashion_mnist_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02O
Mdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOpо
>dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAddBiasAddGdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMul:product:0Udec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2@
>dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAdd
>dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/SigmoidSigmoidGdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2@
>dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/Sigmoid
!tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32#
!tf_op_layer_Maximum_1/Maximum_1/yУ
tf_op_layer_Maximum_1/Maximum_1Maximuminputs*tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_Maximum_1/Maximum_1
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32
tf_op_layer_Maximum/Maximum/yѓ
tf_op_layer_Maximum/MaximumMaximumBdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/Sigmoid:y:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Maximum/Maximumи
 ref_enc_d1/MatMul/ReadVariableOpReadVariableOpQenc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 ref_enc_d1/MatMul/ReadVariableOp
ref_enc_d1/MatMulMatMulinputs(ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ref_enc_d1/MatMulж
!ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpRenc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ref_enc_d1/BiasAdd/ReadVariableOpЎ
ref_enc_d1/BiasAddBiasAddref_enc_d1/MatMul:product:0)ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ref_enc_d1/BiasAdd
tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_1/AddV2_1/yв
tf_op_layer_AddV2_1/AddV2_1AddV2#tf_op_layer_Maximum_1/Maximum_1:z:0&tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_1/AddV2_1{
tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2/AddV2/yТ
tf_op_layer_AddV2/AddV2AddV2tf_op_layer_Maximum/Maximum:z:0"tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2/AddV2z
activation/ReluReluref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
activation/Relu
tf_op_layer_Log/LogLogtf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Log/Log
tf_op_layer_Log_1/Log_1Logtf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Log_1/Log_1р
#ref_z_log_var/MatMul/ReadVariableOpReadVariableOpTenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#ref_z_log_var/MatMul/ReadVariableOpД
ref_z_log_var/MatMulMatMulactivation/Relu:activations:0+ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_log_var/MatMulо
$ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpUenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ref_z_log_var/BiasAdd/ReadVariableOpЙ
ref_z_log_var/BiasAddBiasAddref_z_log_var/MatMul:product:0,ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_log_var/BiasAddз
 ref_z_mean/MatMul/ReadVariableOpReadVariableOpQenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ref_z_mean/MatMul/ReadVariableOpЋ
ref_z_mean/MatMulMatMulactivation/Relu:activations:0(ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_mean/MatMulе
!ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpRenc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ref_z_mean/BiasAdd/ReadVariableOp­
ref_z_mean/BiasAddBiasAddref_z_mean/MatMul:product:0)ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_mean/BiasAddя
/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifferencetf_op_layer_Log/Log:y:0tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ21
/tf_op_layer_SquaredDifference/SquaredDifference
tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_2/AddV2_2/xЬ
tf_op_layer_AddV2_2/AddV2_2AddV2&tf_op_layer_AddV2_2/AddV2_2/x:output:0ref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_2/AddV2_2
tf_op_layer_Square/SquareSquareref_z_mean/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Square/Square
tf_op_layer_Exp/ExpExpref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Exp/Exp
'tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'tf_op_layer_Mean/Mean/reduction_indicesк
tf_op_layer_Mean/MeanMean3tf_op_layer_SquaredDifference/SquaredDifference:z:00tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mean/MeanВ
tf_op_layer_Sub/SubSubtf_op_layer_AddV2_2/AddV2_2:z:0tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sub/Subs
tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD2
tf_op_layer_Mul/Mul/yЎ
tf_op_layer_Mul/MulMultf_op_layer_Mean/Mean:output:0tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mul/MulЌ
tf_op_layer_Sub_1/Sub_1Subtf_op_layer_Sub/Sub:z:0tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sub_1/Sub_1
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%tf_op_layer_Sum/Sum/reduction_indicesЛ
tf_op_layer_Sum/SumSumtf_op_layer_Sub_1/Sub_1:z:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sum/Sum{
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   П2
tf_op_layer_Mul_1/Mul_1/yИ
tf_op_layer_Mul_1/Mul_1Multf_op_layer_Sum/Sum:output:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mul_1/Mul_1Ж
tf_op_layer_AddV2_3/AddV2_3AddV2tf_op_layer_Mul/Mul:z:0tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_3/AddV2_3Є
+tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Mean_1/Mean_1/reduction_indicesХ
tf_op_layer_Mean_1/Mean_1Meantf_op_layer_AddV2_3/AddV2_3:z:04tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mean_1/Mean_1
IdentityIdentityBdec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1Identity"tf_op_layer_Mean_1/Mean_1:output:0*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:џџџџџџџџџ:::::::::::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ч
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13866

inputs-
)decoder_d1_matmul_readvariableop_resource.
*decoder_d1_biasadd_readvariableop_resource1
-decoder_output_matmul_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identityЏ
 decoder_d1/MatMul/ReadVariableOpReadVariableOp)decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 decoder_d1/MatMul/ReadVariableOp
decoder_d1/MatMulMatMulinputs(decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
decoder_d1/MatMulЎ
!decoder_d1/BiasAdd/ReadVariableOpReadVariableOp*decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!decoder_d1/BiasAdd/ReadVariableOpЎ
decoder_d1/BiasAddBiasAdddecoder_d1/MatMul:product:0)decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
decoder_d1/BiasAdd~
activation_1/ReluReludecoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
activation_1/ReluМ
$decoder_output/MatMul/ReadVariableOpReadVariableOp-decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$decoder_output/MatMul/ReadVariableOpК
decoder_output/MatMulMatMulactivation_1/Relu:activations:0,decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
decoder_output/MatMulК
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%decoder_output/BiasAdd/ReadVariableOpО
decoder_output/BiasAddBiasAdddecoder_output/MatMul:product:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
decoder_output/BiasAdd
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
decoder_output/Sigmoido
IdentityIdentitydecoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Н
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12876

inputs
ref_enc_d1_12856
ref_enc_d1_12858
ref_z_mean_12862
ref_z_mean_12864
ref_z_log_var_12867
ref_z_log_var_12869
identity

identity_1

identity_2Ђ"ref_enc_d1/StatefulPartitionedCallЂref_z/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCall
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallinputsref_enc_d1_12856ref_enc_d1_12858*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_126742$
"ref_enc_d1/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_126952
activation/PartitionedCallЛ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_12862ref_z_mean_12864*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_127132$
"ref_z_mean/StatefulPartitionedCallЪ
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_12867ref_z_log_var_12869*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_127392'
%ref_z_log_var/StatefulPartitionedCallЛ
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_127712
ref_z/StatefulPartitionedCall
IdentityIdentity+ref_z_mean/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity&ref_z/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:џџџџџџџџџ::::::2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2>
ref_z/StatefulPartitionedCallref_z/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
х

*__inference_ref_enc_d1_layer_call_fn_13911

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_126742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

H
,__inference_activation_1_layer_call_fn_14021

inputs
identityЩ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_129302
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
о
К
G__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13892

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_130272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
o
@__inference_ref_z_layer_call_and_return_conditional_losses_13986
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
random_normal/stddevЏ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02$
"random_normal/RandomStandardNormalЋ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1

Ч
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13848

inputs-
)decoder_d1_matmul_readvariableop_resource.
*decoder_d1_biasadd_readvariableop_resource1
-decoder_output_matmul_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identityЏ
 decoder_d1/MatMul/ReadVariableOpReadVariableOp)decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 decoder_d1/MatMul/ReadVariableOp
decoder_d1/MatMulMatMulinputs(decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
decoder_d1/MatMulЎ
!decoder_d1/BiasAdd/ReadVariableOpReadVariableOp*decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!decoder_d1/BiasAdd/ReadVariableOpЎ
decoder_d1/BiasAddBiasAdddecoder_d1/MatMul:product:0)decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
decoder_d1/BiasAdd~
activation_1/ReluReludecoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
activation_1/ReluМ
$decoder_output/MatMul/ReadVariableOpReadVariableOp-decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$decoder_output/MatMul/ReadVariableOpК
decoder_output/MatMulMatMulactivation_1/Relu:activations:0,decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
decoder_output/MatMulК
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%decoder_output/BiasAdd/ReadVariableOpО
decoder_output/BiasAddBiasAdddecoder_output/MatMul:product:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
decoder_output/BiasAdd
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
decoder_output/Sigmoido
IdentityIdentitydecoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ:::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
V

b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13330

inputs1
-enc_ref_512_2_msle_g_relu_fashion_mnist_132641
-enc_ref_512_2_msle_g_relu_fashion_mnist_132661
-enc_ref_512_2_msle_g_relu_fashion_mnist_132681
-enc_ref_512_2_msle_g_relu_fashion_mnist_132701
-enc_ref_512_2_msle_g_relu_fashion_mnist_132721
-enc_ref_512_2_msle_g_relu_fashion_mnist_132741
-dec_ref_512_2_msle_g_relu_fashion_mnist_132791
-dec_ref_512_2_msle_g_relu_fashion_mnist_132811
-dec_ref_512_2_msle_g_relu_fashion_mnist_132831
-dec_ref_512_2_msle_g_relu_fashion_mnist_13285
identity

identity_1Ђ?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ"ref_enc_d1/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCall
?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallinputs-enc_ref_512_2_msle_g_relu_fashion_mnist_13264-enc_ref_512_2_msle_g_relu_fashion_mnist_13266-enc_ref_512_2_msle_g_relu_fashion_mnist_13268-enc_ref_512_2_msle_g_relu_fashion_mnist_13270-enc_ref_512_2_msle_g_relu_fashion_mnist_13272-enc_ref_512_2_msle_g_relu_fashion_mnist_13274*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_128322A
?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallд
?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallHenc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:2-dec_ref_512_2_msle_g_relu_fashion_mnist_13279-dec_ref_512_2_msle_g_relu_fashion_mnist_13281-dec_ref_512_2_msle_g_relu_fashion_mnist_13283-dec_ref_512_2_msle_g_relu_fashion_mnist_13285*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_129992A
?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall
!tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32#
!tf_op_layer_Maximum_1/Maximum_1/yУ
tf_op_layer_Maximum_1/Maximum_1Maximuminputs*tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2!
tf_op_layer_Maximum_1/Maximum_1
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32
tf_op_layer_Maximum/Maximum/yљ
tf_op_layer_Maximum/MaximumMaximumHdec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Maximum/Maximumй
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallinputs-enc_ref_512_2_msle_g_relu_fashion_mnist_13264-enc_ref_512_2_msle_g_relu_fashion_mnist_13266*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_126742$
"ref_enc_d1/StatefulPartitionedCall
tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_1/AddV2_1/yв
tf_op_layer_AddV2_1/AddV2_1AddV2#tf_op_layer_Maximum_1/Maximum_1:z:0&tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_1/AddV2_1{
tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2/AddV2/yТ
tf_op_layer_AddV2/AddV2AddV2tf_op_layer_Maximum/Maximum:z:0"tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2/AddV2
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_126952
activation/PartitionedCall
tf_op_layer_Log/LogLogtf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Log/Log
tf_op_layer_Log_1/Log_1Logtf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Log_1/Log_1ў
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0-enc_ref_512_2_msle_g_relu_fashion_mnist_13272-enc_ref_512_2_msle_g_relu_fashion_mnist_13274*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_127392'
%ref_z_log_var/StatefulPartitionedCallѕ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0-enc_ref_512_2_msle_g_relu_fashion_mnist_13268-enc_ref_512_2_msle_g_relu_fashion_mnist_13270*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_127132$
"ref_z_mean/StatefulPartitionedCallя
/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifferencetf_op_layer_Log/Log:y:0tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ21
/tf_op_layer_SquaredDifference/SquaredDifference
tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_AddV2_2/AddV2_2/xм
tf_op_layer_AddV2_2/AddV2_2AddV2&tf_op_layer_AddV2_2/AddV2_2/x:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_2/AddV2_2Ў
tf_op_layer_Square/SquareSquare+ref_z_mean/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Square/SquareЂ
tf_op_layer_Exp/ExpExp.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Exp/Exp
'tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2)
'tf_op_layer_Mean/Mean/reduction_indicesк
tf_op_layer_Mean/MeanMean3tf_op_layer_SquaredDifference/SquaredDifference:z:00tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mean/MeanВ
tf_op_layer_Sub/SubSubtf_op_layer_AddV2_2/AddV2_2:z:0tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sub/Subs
tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD2
tf_op_layer_Mul/Mul/yЎ
tf_op_layer_Mul/MulMultf_op_layer_Mean/Mean:output:0tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mul/MulЌ
tf_op_layer_Sub_1/Sub_1Subtf_op_layer_Sub/Sub:z:0tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sub_1/Sub_1
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2'
%tf_op_layer_Sum/Sum/reduction_indicesЛ
tf_op_layer_Sum/SumSumtf_op_layer_Sub_1/Sub_1:z:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Sum/Sum{
tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   П2
tf_op_layer_Mul_1/Mul_1/yИ
tf_op_layer_Mul_1/Mul_1Multf_op_layer_Sum/Sum:output:0"tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Mul_1/Mul_1Ж
tf_op_layer_AddV2_3/AddV2_3AddV2tf_op_layer_Mul/Mul:z:0tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2
tf_op_layer_AddV2_3/AddV2_3Є
+tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2-
+tf_op_layer_Mean_1/Mean_1/reduction_indicesХ
tf_op_layer_Mean_1/Mean_1Meantf_op_layer_AddV2_3/AddV2_3:z:04tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 2
tf_op_layer_Mean_1/Mean_1ф
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
C__inference_add_loss_layer_call_and_return_conditional_losses_131772
add_loss/PartitionedCall
IdentityIdentityHdec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0@^dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall@^enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityо

Identity_1Identity!add_loss/PartitionedCall:output:1@^dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall@^enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:џџџџџџџџџ::::::::::2
?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall?dec-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall2
?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall?enc-ref-512-2-msle-g-relu-fashion_mnist/StatefulPartitionedCall2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ	

G__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13690

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
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:џџџџџџџџџ: *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_133302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:џџџџџџџџџ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
З
a
E__inference_activation_layer_call_and_return_conditional_losses_12695

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Н
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12832

inputs
ref_enc_d1_12812
ref_enc_d1_12814
ref_z_mean_12818
ref_z_mean_12820
ref_z_log_var_12823
ref_z_log_var_12825
identity

identity_1

identity_2Ђ"ref_enc_d1/StatefulPartitionedCallЂref_z/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCall
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallinputsref_enc_d1_12812ref_enc_d1_12814*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_126742$
"ref_enc_d1/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_126952
activation/PartitionedCallЛ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_12818ref_z_mean_12820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_127132$
"ref_z_mean/StatefulPartitionedCallЪ
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_12823ref_z_log_var_12825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_127392'
%ref_z_log_var/StatefulPartitionedCallЛ
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_127712
ref_z/StatefulPartitionedCall
IdentityIdentity+ref_z_mean/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity&ref_z/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:џџџџџџџџџ::::::2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2>
ref_z/StatefulPartitionedCallref_z/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ	

G__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13716

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
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:џџџџџџџџџ: *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_134252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:џџџџџџџџџ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
И	

G__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13449
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
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:џџџџџџџџџ: *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_134252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:џџџџџџџџџ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameencoder_input

n
%__inference_ref_z_layer_call_fn_13992
inputs_0
inputs_1
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_127712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
ж
ј
G__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13809

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2ЂStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_128322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч
Н
G__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13010
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCall	latent_inunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_129992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	latent_in
З%
Ф
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13788

inputs-
)ref_enc_d1_matmul_readvariableop_resource.
*ref_enc_d1_biasadd_readvariableop_resource-
)ref_z_mean_matmul_readvariableop_resource.
*ref_z_mean_biasadd_readvariableop_resource0
,ref_z_log_var_matmul_readvariableop_resource1
-ref_z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2А
 ref_enc_d1/MatMul/ReadVariableOpReadVariableOp)ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 ref_enc_d1/MatMul/ReadVariableOp
ref_enc_d1/MatMulMatMulinputs(ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ref_enc_d1/MatMulЎ
!ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOp*ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!ref_enc_d1/BiasAdd/ReadVariableOpЎ
ref_enc_d1/BiasAddBiasAddref_enc_d1/MatMul:product:0)ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ref_enc_d1/BiasAddz
activation/ReluReluref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
activation/ReluЏ
 ref_z_mean/MatMul/ReadVariableOpReadVariableOp)ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ref_z_mean/MatMul/ReadVariableOpЋ
ref_z_mean/MatMulMatMulactivation/Relu:activations:0(ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_mean/MatMul­
!ref_z_mean/BiasAdd/ReadVariableOpReadVariableOp*ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ref_z_mean/BiasAdd/ReadVariableOp­
ref_z_mean/BiasAddBiasAddref_z_mean/MatMul:product:0)ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_mean/BiasAddИ
#ref_z_log_var/MatMul/ReadVariableOpReadVariableOp,ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#ref_z_log_var/MatMul/ReadVariableOpД
ref_z_log_var/MatMulMatMulactivation/Relu:activations:0+ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_log_var/MatMulЖ
$ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOp-ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ref_z_log_var/BiasAdd/ReadVariableOpЙ
ref_z_log_var/BiasAddBiasAddref_z_log_var/MatMul:product:0,ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
ref_z/random_normal/stddevС
(ref_z/random_normal/RandomStandardNormalRandomStandardNormalref_z/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02*
(ref_z/random_normal/RandomStandardNormalУ
ref_z/random_normal/mulMul1ref_z/random_normal/RandomStandardNormal:output:0#ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z/random_normal/mulЃ
ref_z/random_normalAddref_z/random_normal/mul:z:0!ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
	ref_z/mul^
	ref_z/ExpExpref_z/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	ref_z/Exp{
ref_z/mul_1Mulref_z/Exp:y:0ref_z/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z/mul_1
	ref_z/addAddV2ref_z_mean/BiasAdd:output:0ref_z/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	ref_z/addo
IdentityIdentityref_z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityv

Identity_1Identityref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1e

Identity_2Identityref_z/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:џџџџџџџџџ:::::::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
В
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12966
	latent_in
decoder_d1_12920
decoder_d1_12922
decoder_output_12960
decoder_output_12962
identityЂ"decoder_d1/StatefulPartitionedCallЂ&decoder_output/StatefulPartitionedCallЂ
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCall	latent_indecoder_d1_12920decoder_d1_12922*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_129092$
"decoder_d1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_129302
activation_1/PartitionedCallв
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_12960decoder_output_12962*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_129492(
&decoder_output/StatefulPartitionedCallв
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	latent_in
ас
г

 __inference__wrapped_model_12660
encoder_input}
yvae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resource~
zvae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resource}
yvae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resource~
zvae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resource
|vae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resource
}vae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resource}
yvae_ref_512_2_msle_g_relu_fashion_mnist_dec_ref_512_2_msle_g_relu_fashion_mnist_decoder_d1_matmul_readvariableop_resource~
zvae_ref_512_2_msle_g_relu_fashion_mnist_dec_ref_512_2_msle_g_relu_fashion_mnist_decoder_d1_biasadd_readvariableop_resource
}vae_ref_512_2_msle_g_relu_fashion_mnist_dec_ref_512_2_msle_g_relu_fashion_mnist_decoder_output_matmul_readvariableop_resource
~vae_ref_512_2_msle_g_relu_fashion_mnist_dec_ref_512_2_msle_g_relu_fashion_mnist_decoder_output_biasadd_readvariableop_resource
identity 
pvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOpReadVariableOpyvae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02r
pvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp
avae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulMatMulencoder_inputxvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2c
avae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul
qvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpzvae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02s
qvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpю
bvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddBiasAddkvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul:product:0yvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2d
bvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddъ
_vae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/activation/ReluRelukvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2a
_vae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/activation/Relu
pvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpReadVariableOpyvae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02r
pvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpы
avae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMulMatMulmvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/activation/Relu:activations:0xvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2c
avae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul
qvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpzvae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02s
qvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpэ
bvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAddBiasAddkvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul:product:0yvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2d
bvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAddЈ
svae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpReadVariableOp|vae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02u
svae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpє
dvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulMatMulmvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/activation/Relu:activations:0{vae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2f
dvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulІ
tvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOp}vae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02v
tvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpљ
evae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddBiasAddnvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul:product:0|vae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2g
evae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddе
[vae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/ShapeShapekvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2]
[vae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/Shape
hvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2j
hvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/mean
jvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2l
jvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/stddevБ
xvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormalRandomStandardNormaldvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02z
xvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormal
gvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/mulMulvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormal:output:0svae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2i
gvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/mulу
cvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normalAddkvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/mul:z:0qvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2e
cvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normalџ
[vae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2]
[vae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul/xХ
Yvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mulMuldvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul/x:output:0nvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2[
Yvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mulЮ
Yvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/ExpExp]vae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2[
Yvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/ExpЛ
[vae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul_1Mul]vae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/Exp:y:0gvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2]
[vae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul_1П
Yvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/addAddV2kvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd:output:0_vae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2[
Yvae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/add
pvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOpReadVariableOpyvae_ref_512_2_msle_g_relu_fashion_mnist_dec_ref_512_2_msle_g_relu_fashion_mnist_decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02r
pvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOpм
avae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMulMatMul]vae-ref-512-2-msle-g-relu-fashion_mnist/enc-ref-512-2-msle-g-relu-fashion_mnist/ref_z/add:z:0xvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2c
avae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMul
qvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOpReadVariableOpzvae_ref_512_2_msle_g_relu_fashion_mnist_dec_ref_512_2_msle_g_relu_fashion_mnist_decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02s
qvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOpю
bvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAddBiasAddkvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/MatMul:product:0yvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2d
bvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAddю
avae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/activation_1/ReluRelukvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2c
avae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/activation_1/ReluЌ
tvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOpReadVariableOp}vae_ref_512_2_msle_g_relu_fashion_mnist_dec_ref_512_2_msle_g_relu_fashion_mnist_decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02v
tvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOpњ
evae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMulMatMulovae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/activation_1/Relu:activations:0|vae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2g
evae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMulЊ
uvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOpReadVariableOp~vae_ref_512_2_msle_g_relu_fashion_mnist_dec_ref_512_2_msle_g_relu_fashion_mnist_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02w
uvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOpў
fvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAddBiasAddovae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/MatMul:product:0}vae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2h
fvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAddџ
fvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/SigmoidSigmoidovae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2h
fvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/Sigmoidл
Ivae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32K
Ivae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Maximum_1/Maximum_1/yТ
Gvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Maximum_1/Maximum_1Maximumencoder_inputRvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2I
Gvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Maximum_1/Maximum_1г
Evae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32G
Evae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Maximum/Maximum/y
Cvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Maximum/MaximumMaximumjvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/Sigmoid:y:0Nvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2E
Cvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Maximum/Maximumа
Hvae-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOpReadVariableOpyvae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02J
Hvae-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp
9vae-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulMatMulencoder_inputPvae-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2;
9vae-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulЮ
Ivae-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpzvae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02K
Ivae-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpЮ
:vae-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddBiasAddCvae-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul:product:0Qvae-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2<
:vae-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddг
Evae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2G
Evae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_1/AddV2_1/yђ
Cvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_1/AddV2_1AddV2Kvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Maximum_1/Maximum_1:z:0Nvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2E
Cvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_1/AddV2_1Ы
Avae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2C
Avae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2/AddV2/yт
?vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2/AddV2AddV2Gvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Maximum/Maximum:z:0Jvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2A
?vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2/AddV2ђ
7vae-ref-512-2-msle-g-relu-fashion_mnist/activation/ReluReluCvae-ref-512-2-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ29
7vae-ref-512-2-msle-g-relu-fashion_mnist/activation/Relu
;vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Log/LogLogCvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2=
;vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Log/Log
?vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Log_1/Log_1LogGvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2A
?vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Log_1/Log_1и
Kvae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpReadVariableOp|vae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02M
Kvae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpд
<vae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulMatMulEvae-ref-512-2-msle-g-relu-fashion_mnist/activation/Relu:activations:0Svae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2>
<vae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulж
Lvae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOp}vae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02N
Lvae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpй
=vae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddBiasAddFvae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul:product:0Tvae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2?
=vae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddЯ
Hvae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpReadVariableOpyvae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02J
Hvae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpЫ
9vae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMulMatMulEvae-ref-512-2-msle-g-relu-fashion_mnist/activation/Relu:activations:0Pvae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2;
9vae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMulЭ
Ivae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpzvae_ref_512_2_msle_g_relu_fashion_mnist_enc_ref_512_2_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02K
Ivae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpЭ
:vae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAddBiasAddCvae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/MatMul:product:0Qvae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2<
:vae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd
Wvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifference?vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Log/Log:y:0Cvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2Y
Wvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_SquaredDifference/SquaredDifferenceг
Evae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2G
Evae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_2/AddV2_2/xь
Cvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_2/AddV2_2AddV2Nvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_2/AddV2_2/x:output:0Fvae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2E
Cvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_2/AddV2_2
Avae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Square/SquareSquareCvae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2C
Avae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Square/Square
;vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Exp/ExpExpFvae-ref-512-2-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2=
;vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Exp/Expэ
Ovae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2Q
Ovae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mean/Mean/reduction_indicesњ
=vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mean/MeanMean[vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_SquaredDifference/SquaredDifference:z:0Xvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2?
=vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mean/Meanв
;vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Sub/SubSubGvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_2/AddV2_2:z:0Evae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2=
;vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Sub/SubУ
=vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD2?
=vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mul/Mul/yЮ
;vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mul/MulMulFvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mean/Mean:output:0Fvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2=
;vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mul/MulЬ
?vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Sub_1/Sub_1Sub?vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Sub/Sub:z:0?vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2A
?vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Sub_1/Sub_1щ
Mvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2O
Mvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Sum/Sum/reduction_indicesл
;vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Sum/SumSumCvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Sub_1/Sub_1:z:0Vvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2=
;vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Sum/SumЫ
Avae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   П2C
Avae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mul_1/Mul_1/yи
?vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mul_1/Mul_1MulDvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Sum/Sum:output:0Jvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2A
?vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mul_1/Mul_1ж
Cvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_3/AddV2_3AddV2?vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mul/Mul:z:0Cvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2E
Cvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_3/AddV2_3є
Svae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2U
Svae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mean_1/Mean_1/reduction_indicesх
Avae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mean_1/Mean_1MeanGvae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_3/AddV2_3:z:0\vae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 2C
Avae-ref-512-2-msle-g-relu-fashion_mnist/tf_op_layer_Mean_1/Mean_1П
IdentityIdentityjvae-ref-512-2-msle-g-relu-fashion_mnist/dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:џџџџџџџџџ:::::::::::W S
(
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameencoder_input
З
a
E__inference_activation_layer_call_and_return_conditional_losses_13916

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
о
К
G__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13879

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *k
ffRd
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_129992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"ФL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*и
serving_defaultФ
H
encoder_input7
serving_default_encoder_input:0џџџџџџџџџ\
'dec-ref-512-2-msle-g-relu-fashion_mnist1
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:іы
Яд
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
	variables
regularization_losses
trainable_variables
 	keras_api
!
signatures
Щ__call__
Ъ_default_save_signature
+Ы&call_and_return_all_conditional_losses"Я
_tf_keras_networkѕЮ{"class_name": "Functional", "name": "vae-ref-512-2-msle-g-relu-fashion_mnist", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "vae-ref-512-2-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "enc-ref-512-2-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}, "name": "ref_z", "inbound_nodes": [[["ref_z_mean", 0, 0, {}], ["ref_z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["ref_z_mean", 0, 0], ["ref_z_log_var", 0, 0], ["ref_z", 0, 0]]}, "name": "enc-ref-512-2-msle-g-relu-fashion_mnist", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "dec-ref-512-2-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "name": "dec-ref-512-2-msle-g-relu-fashion_mnist", "inbound_nodes": [[["enc-ref-512-2-msle-g-relu-fashion_mnist", 1, 2, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/Sigmoid", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}, "name": "tf_op_layer_Maximum", "inbound_nodes": [[["dec-ref-512-2-msle-g-relu-fashion_mnist", 1, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum_1", "op": "Maximum", "input": ["encoder_input", "Maximum_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}, "name": "tf_op_layer_Maximum_1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2", "op": "AddV2", "input": ["Maximum", "AddV2/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_AddV2", "inbound_nodes": [[["tf_op_layer_Maximum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_1", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_1", "op": "AddV2", "input": ["Maximum_1", "AddV2_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_AddV2_1", "inbound_nodes": [[["tf_op_layer_Maximum_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_2", "op": "AddV2", "input": ["AddV2_2/x", "ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 1.0}}, "name": "tf_op_layer_AddV2_2", "inbound_nodes": [[["ref_z_log_var", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Square", "trainable": true, "dtype": "float32", "node_def": {"name": "Square", "op": "Square", "input": ["ref_z_mean/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Square", "inbound_nodes": [[["ref_z_mean", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Log", "trainable": true, "dtype": "float32", "node_def": {"name": "Log", "op": "Log", "input": ["AddV2"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Log", "inbound_nodes": [[["tf_op_layer_AddV2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Log_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Log_1", "op": "Log", "input": ["AddV2_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Log_1", "inbound_nodes": [[["tf_op_layer_AddV2_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["AddV2_2", "Square"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub", "inbound_nodes": [[["tf_op_layer_AddV2_2", 0, 0, {}], ["tf_op_layer_Square", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Exp", "trainable": true, "dtype": "float32", "node_def": {"name": "Exp", "op": "Exp", "input": ["ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Exp", "inbound_nodes": [[["ref_z_log_var", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "SquaredDifference", "trainable": true, "dtype": "float32", "node_def": {"name": "SquaredDifference", "op": "SquaredDifference", "input": ["Log", "Log_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_SquaredDifference", "inbound_nodes": [[["tf_op_layer_Log", 0, 0, {}], ["tf_op_layer_Log_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub_1", "op": "Sub", "input": ["Sub", "Exp"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub_1", "inbound_nodes": [[["tf_op_layer_Sub", 0, 0, {}], ["tf_op_layer_Exp", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean", "op": "Mean", "input": ["SquaredDifference", "Mean/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Mean", "inbound_nodes": [[["tf_op_layer_SquaredDifference", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["Sub_1", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Sum", "inbound_nodes": [[["tf_op_layer_Sub_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["Mean", "Mul/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 784.0}}, "name": "tf_op_layer_Mul", "inbound_nodes": [[["tf_op_layer_Mean", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["Sum", "Mul_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": -0.5}}, "name": "tf_op_layer_Mul_1", "inbound_nodes": [[["tf_op_layer_Sum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_3", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_3", "op": "AddV2", "input": ["Mul", "Mul_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_3", "inbound_nodes": [[["tf_op_layer_Mul", 0, 0, {}], ["tf_op_layer_Mul_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean_1", "op": "Mean", "input": ["AddV2_3", "Mean_1/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Mean_1", "inbound_nodes": [[["tf_op_layer_AddV2_3", 0, 0, {}]]]}, {"class_name": "AddLoss", "config": {"name": "add_loss", "trainable": true, "dtype": "float32", "unconditional": false}, "name": "add_loss", "inbound_nodes": [[["tf_op_layer_Mean_1", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["dec-ref-512-2-msle-g-relu-fashion_mnist", 1, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "vae-ref-512-2-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "enc-ref-512-2-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}, "name": "ref_z", "inbound_nodes": [[["ref_z_mean", 0, 0, {}], ["ref_z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["ref_z_mean", 0, 0], ["ref_z_log_var", 0, 0], ["ref_z", 0, 0]]}, "name": "enc-ref-512-2-msle-g-relu-fashion_mnist", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "dec-ref-512-2-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "name": "dec-ref-512-2-msle-g-relu-fashion_mnist", "inbound_nodes": [[["enc-ref-512-2-msle-g-relu-fashion_mnist", 1, 2, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/Sigmoid", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}, "name": "tf_op_layer_Maximum", "inbound_nodes": [[["dec-ref-512-2-msle-g-relu-fashion_mnist", 1, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum_1", "op": "Maximum", "input": ["encoder_input", "Maximum_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}, "name": "tf_op_layer_Maximum_1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2", "op": "AddV2", "input": ["Maximum", "AddV2/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_AddV2", "inbound_nodes": [[["tf_op_layer_Maximum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_1", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_1", "op": "AddV2", "input": ["Maximum_1", "AddV2_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_AddV2_1", "inbound_nodes": [[["tf_op_layer_Maximum_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_2", "op": "AddV2", "input": ["AddV2_2/x", "ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 1.0}}, "name": "tf_op_layer_AddV2_2", "inbound_nodes": [[["ref_z_log_var", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Square", "trainable": true, "dtype": "float32", "node_def": {"name": "Square", "op": "Square", "input": ["ref_z_mean/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Square", "inbound_nodes": [[["ref_z_mean", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Log", "trainable": true, "dtype": "float32", "node_def": {"name": "Log", "op": "Log", "input": ["AddV2"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Log", "inbound_nodes": [[["tf_op_layer_AddV2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Log_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Log_1", "op": "Log", "input": ["AddV2_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Log_1", "inbound_nodes": [[["tf_op_layer_AddV2_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["AddV2_2", "Square"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub", "inbound_nodes": [[["tf_op_layer_AddV2_2", 0, 0, {}], ["tf_op_layer_Square", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Exp", "trainable": true, "dtype": "float32", "node_def": {"name": "Exp", "op": "Exp", "input": ["ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Exp", "inbound_nodes": [[["ref_z_log_var", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "SquaredDifference", "trainable": true, "dtype": "float32", "node_def": {"name": "SquaredDifference", "op": "SquaredDifference", "input": ["Log", "Log_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_SquaredDifference", "inbound_nodes": [[["tf_op_layer_Log", 0, 0, {}], ["tf_op_layer_Log_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub_1", "op": "Sub", "input": ["Sub", "Exp"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub_1", "inbound_nodes": [[["tf_op_layer_Sub", 0, 0, {}], ["tf_op_layer_Exp", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean", "op": "Mean", "input": ["SquaredDifference", "Mean/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Mean", "inbound_nodes": [[["tf_op_layer_SquaredDifference", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["Sub_1", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Sum", "inbound_nodes": [[["tf_op_layer_Sub_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["Mean", "Mul/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 784.0}}, "name": "tf_op_layer_Mul", "inbound_nodes": [[["tf_op_layer_Mean", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["Sum", "Mul_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": -0.5}}, "name": "tf_op_layer_Mul_1", "inbound_nodes": [[["tf_op_layer_Sum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_3", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_3", "op": "AddV2", "input": ["Mul", "Mul_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_3", "inbound_nodes": [[["tf_op_layer_Mul", 0, 0, {}], ["tf_op_layer_Mul_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean_1", "op": "Mean", "input": ["AddV2_3", "Mean_1/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Mean_1", "inbound_nodes": [[["tf_op_layer_AddV2_3", 0, 0, {}]]]}, {"class_name": "AddLoss", "config": {"name": "add_loss", "trainable": true, "dtype": "float32", "unconditional": false}, "name": "add_loss", "inbound_nodes": [[["tf_op_layer_Mean_1", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["dec-ref-512-2-msle-g-relu-fashion_mnist", 1, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
љ"і
_tf_keras_input_layerж{"class_name": "InputLayer", "name": "encoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}}
*
layer-0
layer_with_weights-0
layer-1
layer-2
	layer_with_weights-1
	layer-3
layer_with_weights-2
layer-4
"layer-5
#	variables
$regularization_losses
%trainable_variables
&	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses"н'
_tf_keras_networkС'{"class_name": "Functional", "name": "enc-ref-512-2-msle-g-relu-fashion_mnist", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "enc-ref-512-2-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}, "name": "ref_z", "inbound_nodes": [[["ref_z_mean", 0, 0, {}], ["ref_z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["ref_z_mean", 0, 0], ["ref_z_log_var", 0, 0], ["ref_z", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "enc-ref-512-2-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}, "name": "ref_z", "inbound_nodes": [[["ref_z_mean", 0, 0, {}], ["ref_z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["ref_z_mean", 0, 0], ["ref_z_log_var", 0, 0], ["ref_z", 0, 0]]}}}

'layer-0
(layer_with_weights-0
(layer-1
)layer-2
*layer_with_weights-1
*layer-3
+	variables
,regularization_losses
-trainable_variables
.	keras_api
Ю__call__
+Я&call_and_return_all_conditional_losses"Ѕ
_tf_keras_network{"class_name": "Functional", "name": "dec-ref-512-2-msle-g-relu-fashion_mnist", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "dec-ref-512-2-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "dec-ref-512-2-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}}}
§

/kernel
0bias
1	variables
2regularization_losses
3trainable_variables
4	keras_api
а__call__
+б&call_and_return_all_conditional_losses"ж
_tf_keras_layerМ{"class_name": "Dense", "name": "ref_enc_d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
г
5	variables
6regularization_losses
7trainable_variables
8	keras_api
в__call__
+г&call_and_return_all_conditional_losses"Т
_tf_keras_layerЈ{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
Ї
9	keras_api"
_tf_keras_layerћ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Maximum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["dec-ref-512-2-msle-g-relu-fashion_mnist/decoder_output/Sigmoid", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}}
ў
:	keras_api"ь
_tf_keras_layerв{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Maximum_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Maximum_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum_1", "op": "Maximum", "input": ["encoder_input", "Maximum_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}}


;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
д__call__
+е&call_and_return_all_conditional_losses"к
_tf_keras_layerР{"class_name": "Dense", "name": "ref_z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
ћ

Akernel
Bbias
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"д
_tf_keras_layerК{"class_name": "Dense", "name": "ref_z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
г
G	keras_api"С
_tf_keras_layerЇ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "AddV2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2", "op": "AddV2", "input": ["Maximum", "AddV2/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}}
н
H	keras_api"Ы
_tf_keras_layerБ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "AddV2_1", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_1", "op": "AddV2", "input": ["Maximum_1", "AddV2_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}}
щ
I	keras_api"з
_tf_keras_layerН{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "AddV2_2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_2", "op": "AddV2", "input": ["AddV2_2/x", "ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 1.0}}}
Я
J	keras_api"Н
_tf_keras_layerЃ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Square", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Square", "trainable": true, "dtype": "float32", "node_def": {"name": "Square", "op": "Square", "input": ["ref_z_mean/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
Ж
K	keras_api"Є
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Log", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Log", "trainable": true, "dtype": "float32", "node_def": {"name": "Log", "op": "Log", "input": ["AddV2"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
О
L	keras_api"Ќ
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Log_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Log_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Log_1", "op": "Log", "input": ["AddV2_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
Т
M	keras_api"А
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sub", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["AddV2_2", "Square"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
Ц
N	keras_api"Д
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Exp", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Exp", "trainable": true, "dtype": "float32", "node_def": {"name": "Exp", "op": "Exp", "input": ["ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
ѕ
O	keras_api"у
_tf_keras_layerЩ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_SquaredDifference", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "SquaredDifference", "trainable": true, "dtype": "float32", "node_def": {"name": "SquaredDifference", "op": "SquaredDifference", "input": ["Log", "Log_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
С
P	keras_api"Џ
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sub_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Sub_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub_1", "op": "Sub", "input": ["Sub", "Exp"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
 
Q	keras_api"
_tf_keras_layerє{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Mean", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean", "op": "Mean", "input": ["SquaredDifference", "Mean/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}}

R	keras_api"§
_tf_keras_layerу{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["Sub_1", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}}
Ш
S	keras_api"Ж
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Mul", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["Mean", "Mul/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 784.0}}}
Ю
T	keras_api"М
_tf_keras_layerЂ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["Sum", "Mul_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": -0.5}}}
Ы
U	keras_api"Й
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "AddV2_3", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_3", "op": "AddV2", "input": ["Mul", "Mul_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}

V	keras_api"
_tf_keras_layerѓ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mean_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Mean_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean_1", "op": "Mean", "input": ["AddV2_3", "Mean_1/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0]}}}
Ю
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
и__call__
+й&call_and_return_all_conditional_losses"Н
_tf_keras_layerЃ{"class_name": "AddLoss", "name": "add_loss", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_loss", "trainable": true, "dtype": "float32", "unconditional": false}}

[iter

\beta_1

]beta_2
	^decay
_learning_rate/mЕ0mЖ;mЗ<mИAmЙBmК`mЛamМbmНcmО/vП0vР;vС<vТAvУBvФ`vХavЦbvЧcvШ"
	optimizer
 "
trackable_dict_wrapper
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
Ю
dlayer_regularization_losses
	variables
enon_trainable_variables
regularization_losses
fmetrics

glayers
trainable_variables
hlayer_metrics
Щ__call__
Ъ_default_save_signature
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
-
кserving_default"
signature_map
Б
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
л__call__
+м&call_and_return_all_conditional_losses" 
_tf_keras_layer{"class_name": "Sampling", "name": "ref_z", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}}
J
/0
01
A2
B3
;4
<5"
trackable_list_wrapper
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
А
mlayer_regularization_losses
#	variables
nnon_trainable_variables
$regularization_losses
ometrics

players
%trainable_variables
qlayer_metrics
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
э"ъ
_tf_keras_input_layerЪ{"class_name": "InputLayer", "name": "latent_in", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}}
љ

`kernel
abias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
н__call__
+о&call_and_return_all_conditional_losses"в
_tf_keras_layerИ{"class_name": "Dense", "name": "decoder_d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
з
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
п__call__
+р&call_and_return_all_conditional_losses"Ц
_tf_keras_layerЌ{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}


bkernel
cbias
z	variables
{regularization_losses
|trainable_variables
}	keras_api
с__call__
+т&call_and_return_all_conditional_losses"п
_tf_keras_layerХ{"class_name": "Dense", "name": "decoder_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
<
`0
a1
b2
c3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
`0
a1
b2
c3"
trackable_list_wrapper
Г
~layer_regularization_losses
+	variables
non_trainable_variables
,regularization_losses
metrics
layers
-trainable_variables
layer_metrics
Ю__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
%:#
2ref_enc_d1/kernel
:2ref_enc_d1/bias
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
Е
 layer_regularization_losses
1	variables
non_trainable_variables
2regularization_losses
metrics
layers
3trainable_variables
layer_metrics
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 layer_regularization_losses
5	variables
non_trainable_variables
6regularization_losses
metrics
layers
7trainable_variables
layer_metrics
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
':%	2ref_z_log_var/kernel
 :2ref_z_log_var/bias
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
Е
 layer_regularization_losses
=	variables
non_trainable_variables
>regularization_losses
metrics
layers
?trainable_variables
layer_metrics
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
$:"	2ref_z_mean/kernel
:2ref_z_mean/bias
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
Е
 layer_regularization_losses
C	variables
non_trainable_variables
Dregularization_losses
metrics
layers
Etrainable_variables
layer_metrics
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
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
Е
 layer_regularization_losses
W	variables
non_trainable_variables
Xregularization_losses
metrics
layers
Ytrainable_variables
layer_metrics
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
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
(
0"
trackable_list_wrapper
ц
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 layer_regularization_losses
i	variables
non_trainable_variables
jregularization_losses
metrics
 layers
ktrainable_variables
Ёlayer_metrics
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
trackable_dict_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
Е
 Ђlayer_regularization_losses
r	variables
Ѓnon_trainable_variables
sregularization_losses
Єmetrics
Ѕlayers
ttrainable_variables
Іlayer_metrics
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 Їlayer_regularization_losses
v	variables
Јnon_trainable_variables
wregularization_losses
Љmetrics
Њlayers
xtrainable_variables
Ћlayer_metrics
п__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
Е
 Ќlayer_regularization_losses
z	variables
­non_trainable_variables
{regularization_losses
Ўmetrics
Џlayers
|trainable_variables
Аlayer_metrics
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
'0
(1
)2
*3"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
П

Бtotal

Вcount
Г	variables
Д	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
:  (2total
:  (2count
0
Б0
В1"
trackable_list_wrapper
.
Г	variables"
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
ъ2ч
G__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13716
G__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13690
G__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13449
G__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13354Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
х2т
 __inference__wrapped_model_12660Н
В
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
annotationsЊ *-Ђ*
(%
encoder_inputџџџџџџџџџ
ж2г
b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13574
b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13664
b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13189
b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13258Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
G__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13809
G__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13830
G__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_12851
G__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_12895Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13788
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12783
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13752
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12806Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
G__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13010
G__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13879
G__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13892
G__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13038Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ж2г
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12966
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12981
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13866
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13848Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
д2б
*__inference_ref_enc_d1_layer_call_fn_13911Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_13902Ђ
В
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
annotationsЊ *
 
д2б
*__inference_activation_layer_call_fn_13921Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_activation_layer_call_and_return_conditional_losses_13916Ђ
В
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
annotationsЊ *
 
з2д
-__inference_ref_z_log_var_layer_call_fn_13940Ђ
В
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
annotationsЊ *
 
ђ2я
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_13931Ђ
В
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
annotationsЊ *
 
д2б
*__inference_ref_z_mean_layer_call_fn_13959Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_13950Ђ
В
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
annotationsЊ *
 
в2Я
(__inference_add_loss_layer_call_fn_13970Ђ
В
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
annotationsЊ *
 
э2ъ
C__inference_add_loss_layer_call_and_return_conditional_losses_13964Ђ
В
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
annotationsЊ *
 
8B6
#__inference_signature_wrapper_13484encoder_input
Я2Ь
%__inference_ref_z_layer_call_fn_13992Ђ
В
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
annotationsЊ *
 
ъ2ч
@__inference_ref_z_layer_call_and_return_conditional_losses_13986Ђ
В
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
annotationsЊ *
 
д2б
*__inference_decoder_d1_layer_call_fn_14011Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_decoder_d1_layer_call_and_return_conditional_losses_14002Ђ
В
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
annotationsЊ *
 
ж2г
,__inference_activation_1_layer_call_fn_14021Ђ
В
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
annotationsЊ *
 
ё2ю
G__inference_activation_1_layer_call_and_return_conditional_losses_14016Ђ
В
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
annotationsЊ *
 
и2е
.__inference_decoder_output_layer_call_fn_14041Ђ
В
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
annotationsЊ *
 
ѓ2№
I__inference_decoder_output_layer_call_and_return_conditional_losses_14032Ђ
В
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
annotationsЊ *
 о
 __inference__wrapped_model_12660Й
/0AB;<`abc7Ђ4
-Ђ*
(%
encoder_inputџџџџџџџџџ
Њ "rЊo
m
'dec-ref-512-2-msle-g-relu-fashion_mnistB?
'dec-ref-512-2-msle-g-relu-fashion_mnistџџџџџџџџџЅ
G__inference_activation_1_layer_call_and_return_conditional_losses_14016Z0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 }
,__inference_activation_1_layer_call_fn_14021M0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЃ
E__inference_activation_layer_call_and_return_conditional_losses_13916Z0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 {
*__inference_activation_layer_call_fn_13921M0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ
C__inference_add_loss_layer_call_and_return_conditional_losses_13964DЂ
Ђ

inputs 
Њ ""Ђ


0 

	
1/0 U
(__inference_add_loss_layer_call_fn_13970)Ђ
Ђ

inputs 
Њ " а
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12966j`abc:Ђ7
0Ђ-
# 
	latent_inџџџџџџџџџ
p

 
Њ "&Ђ#

0џџџџџџџџџ
 а
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12981j`abc:Ђ7
0Ђ-
# 
	latent_inџџџџџџџџџ
p 

 
Њ "&Ђ#

0џџџџџџџџџ
 Э
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13848g`abc7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "&Ђ#

0џџџџџџџџџ
 Э
b__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13866g`abc7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "&Ђ#

0џџџџџџџџџ
 Ј
G__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13010]`abc:Ђ7
0Ђ-
# 
	latent_inџџџџџџџџџ
p

 
Њ "џџџџџџџџџЈ
G__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13038]`abc:Ђ7
0Ђ-
# 
	latent_inџџџџџџџџџ
p 

 
Њ "џџџџџџџџџЅ
G__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13879Z`abc7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЅ
G__inference_dec-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13892Z`abc7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџІ
E__inference_decoder_d1_layer_call_and_return_conditional_losses_14002]`a/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 ~
*__inference_decoder_d1_layer_call_fn_14011P`a/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЋ
I__inference_decoder_output_layer_call_and_return_conditional_losses_14032^bc0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
.__inference_decoder_output_layer_call_fn_14041Qbc0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12783Е/0AB;<?Ђ<
5Ђ2
(%
encoder_inputџџџџџџџџџ
p

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_12806Е/0AB;<?Ђ<
5Ђ2
(%
encoder_inputџџџџџџџџџ
p 

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13752Ў/0AB;<8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 
b__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13788Ў/0AB;<8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p 

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 ё
G__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_12851Ѕ/0AB;<?Ђ<
5Ђ2
(%
encoder_inputџџџџџџџџџ
p

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџё
G__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_12895Ѕ/0AB;<?Ђ<
5Ђ2
(%
encoder_inputџџџџџџџџџ
p 

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџъ
G__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13809/0AB;<8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџъ
G__inference_enc-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13830/0AB;<8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p 

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџЇ
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_13902^/00Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_ref_enc_d1_layer_call_fn_13911Q/00Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџШ
@__inference_ref_z_layer_call_and_return_conditional_losses_13986ZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
%__inference_ref_z_layer_call_fn_13992vZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "џџџџџџџџџЉ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_13931];<0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
-__inference_ref_z_log_var_layer_call_fn_13940P;<0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_13950]AB0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
*__inference_ref_z_mean_layer_call_fn_13959PAB0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџђ
#__inference_signature_wrapper_13484Ъ
/0AB;<`abcHЂE
Ђ 
>Њ;
9
encoder_input(%
encoder_inputџџџџџџџџџ"rЊo
m
'dec-ref-512-2-msle-g-relu-fashion_mnistB?
'dec-ref-512-2-msle-g-relu-fashion_mnistџџџџџџџџџъ
b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13189
/0AB;<`abc?Ђ<
5Ђ2
(%
encoder_inputџџџџџџџџџ
p

 
Њ "4Ђ1

0џџџџџџџџџ

	
1/0 ъ
b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13258
/0AB;<`abc?Ђ<
5Ђ2
(%
encoder_inputџџџџџџџџџ
p 

 
Њ "4Ђ1

0џџџџџџџџџ

	
1/0 т
b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13574|
/0AB;<`abc8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p

 
Њ "4Ђ1

0џџџџџџџџџ

	
1/0 т
b__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_13664|
/0AB;<`abc8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p 

 
Њ "4Ђ1

0џџџџџџџџџ

	
1/0 Г
G__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13354h
/0AB;<`abc?Ђ<
5Ђ2
(%
encoder_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџГ
G__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13449h
/0AB;<`abc?Ђ<
5Ђ2
(%
encoder_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџЌ
G__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13690a
/0AB;<`abc8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЌ
G__inference_vae-ref-512-2-msle-g-relu-fashion_mnist_layer_call_fn_13716a
/0AB;<`abc8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ