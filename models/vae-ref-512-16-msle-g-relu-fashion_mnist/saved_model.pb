ћс
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
 "serve*2.4.0-dev202007052v1.12.1-35711-g8202ae9d5c8­э
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
shape:	*%
shared_nameref_z_log_var/kernel
~
(ref_z_log_var/kernel/Read/ReadVariableOpReadVariableOpref_z_log_var/kernel*
_output_shapes
:	*
dtype0
|
ref_z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameref_z_log_var/bias
u
&ref_z_log_var/bias/Read/ReadVariableOpReadVariableOpref_z_log_var/bias*
_output_shapes
:*
dtype0

ref_z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_nameref_z_mean/kernel
x
%ref_z_mean/kernel/Read/ReadVariableOpReadVariableOpref_z_mean/kernel*
_output_shapes
:	*
dtype0
v
ref_z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameref_z_mean/bias
o
#ref_z_mean/bias/Read/ReadVariableOpReadVariableOpref_z_mean/bias*
_output_shapes
:*
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
shape:	*"
shared_namedecoder_d1/kernel
x
%decoder_d1/kernel/Read/ReadVariableOpReadVariableOpdecoder_d1/kernel*
_output_shapes
:	*
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
shape:	*,
shared_nameAdam/ref_z_log_var/kernel/m

/Adam/ref_z_log_var/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ref_z_log_var/kernel/m*
_output_shapes
:	*
dtype0

Adam/ref_z_log_var/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/ref_z_log_var/bias/m

-Adam/ref_z_log_var/bias/m/Read/ReadVariableOpReadVariableOpAdam/ref_z_log_var/bias/m*
_output_shapes
:*
dtype0

Adam/ref_z_mean/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/ref_z_mean/kernel/m

,Adam/ref_z_mean/kernel/m/Read/ReadVariableOpReadVariableOpAdam/ref_z_mean/kernel/m*
_output_shapes
:	*
dtype0

Adam/ref_z_mean/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/ref_z_mean/bias/m
}
*Adam/ref_z_mean/bias/m/Read/ReadVariableOpReadVariableOpAdam/ref_z_mean/bias/m*
_output_shapes
:*
dtype0

Adam/decoder_d1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/decoder_d1/kernel/m

,Adam/decoder_d1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoder_d1/kernel/m*
_output_shapes
:	*
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
shape:	*,
shared_nameAdam/ref_z_log_var/kernel/v

/Adam/ref_z_log_var/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ref_z_log_var/kernel/v*
_output_shapes
:	*
dtype0

Adam/ref_z_log_var/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/ref_z_log_var/bias/v

-Adam/ref_z_log_var/bias/v/Read/ReadVariableOpReadVariableOpAdam/ref_z_log_var/bias/v*
_output_shapes
:*
dtype0

Adam/ref_z_mean/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/ref_z_mean/kernel/v

,Adam/ref_z_mean/kernel/v/Read/ReadVariableOpReadVariableOpAdam/ref_z_mean/kernel/v*
_output_shapes
:	*
dtype0

Adam/ref_z_mean/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/ref_z_mean/bias/v
}
*Adam/ref_z_mean/bias/v/Read/ReadVariableOpReadVariableOpAdam/ref_z_mean/bias/v*
_output_shapes
:*
dtype0

Adam/decoder_d1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/decoder_d1/kernel/v

,Adam/decoder_d1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoder_d1/kernel/v*
_output_shapes
:	*
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
УL
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ўK
valueєKBёK BъK
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
trainable_variables
regularization_losses
	variables
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
#trainable_variables
$regularization_losses
%	variables
&	keras_api
К
'layer-0
(layer_with_weights-0
(layer-1
)layer-2
*layer_with_weights-1
*layer-3
+trainable_variables
,regularization_losses
-	variables
.	keras_api
h

/kernel
0bias
1trainable_variables
2regularization_losses
3	variables
4	keras_api
R
5trainable_variables
6regularization_losses
7	variables
8	keras_api

9	keras_api

:	keras_api
h

;kernel
<bias
=trainable_variables
>regularization_losses
?	variables
@	keras_api
h

Akernel
Bbias
Ctrainable_variables
Dregularization_losses
E	variables
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
Wtrainable_variables
Xregularization_losses
Y	variables
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

elayers
trainable_variables
flayer_metrics
gnon_trainable_variables
regularization_losses
hmetrics
	variables
 
R
itrainable_variables
jregularization_losses
k	variables
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

nlayers
#trainable_variables
olayer_metrics
pnon_trainable_variables
$regularization_losses
qmetrics
%	variables
 
h

`kernel
abias
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
R
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
h

bkernel
cbias
ztrainable_variables
{regularization_losses
|	variables
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

layers
+trainable_variables
layer_metrics
non_trainable_variables
,regularization_losses
metrics
-	variables
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
layers
layer_metrics
1trainable_variables
non_trainable_variables
2regularization_losses
metrics
3	variables
 
 
 
В
 layer_regularization_losses
layers
layer_metrics
5trainable_variables
non_trainable_variables
6regularization_losses
metrics
7	variables
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
layers
layer_metrics
=trainable_variables
non_trainable_variables
>regularization_losses
metrics
?	variables
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
layers
layer_metrics
Ctrainable_variables
non_trainable_variables
Dregularization_losses
metrics
E	variables
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
layers
layer_metrics
Wtrainable_variables
non_trainable_variables
Xregularization_losses
metrics
Y	variables
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
WU
VARIABLE_VALUEdecoder_d1/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdecoder_d1/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEdecoder_output/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdecoder_output/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
 
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

0
 
 
 
В
 layer_regularization_losses
layers
layer_metrics
itrainable_variables
 non_trainable_variables
jregularization_losses
Ёmetrics
k	variables
 
*
0
1
2
	3
4
"5
 
 
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
Ѓlayers
Єlayer_metrics
rtrainable_variables
Ѕnon_trainable_variables
sregularization_losses
Іmetrics
t	variables
 
 
 
В
 Їlayer_regularization_losses
Јlayers
Љlayer_metrics
vtrainable_variables
Њnon_trainable_variables
wregularization_losses
Ћmetrics
x	variables

b0
c1
 

b0
c1
В
 Ќlayer_regularization_losses
­layers
Ўlayer_metrics
ztrainable_variables
Џnon_trainable_variables
{regularization_losses
Аmetrics
|	variables
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
zx
VARIABLE_VALUEAdam/decoder_d1/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/decoder_d1/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/decoder_output/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/decoder_output/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
zx
VARIABLE_VALUEAdam/decoder_d1/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/decoder_d1/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/decoder_output/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/decoder_output/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
#__inference_signature_wrapper_15535
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
__inference__traced_save_16226
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
!__inference__traced_restore_16347ЂЙ
Є
Х
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_14857
encoder_input
ref_enc_d1_14837
ref_enc_d1_14839
ref_z_mean_14843
ref_z_mean_14845
ref_z_log_var_14848
ref_z_log_var_14850
identity

identity_1

identity_2Ђ"ref_enc_d1/StatefulPartitionedCallЂref_z/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCallІ
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputref_enc_d1_14837ref_enc_d1_14839*
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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_147462
activation/PartitionedCallЛ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_14843ref_z_mean_14845*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
"ref_z_mean/StatefulPartitionedCallЪ
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_14848ref_z_log_var_14850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCallЛ
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
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
:џџџџџџџџџ2

Identity

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity&ref_z/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

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
р
Л
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15943

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallБ
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
GPU2*0J 8 *l
fgRe
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_150782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є
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
г
­
E__inference_decoder_d1_layer_call_and_return_conditional_losses_16053

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
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
:џџџџџџџџџ:::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ьV

c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15309
encoder_input2
.enc_ref_512_16_msle_g_relu_fashion_mnist_152432
.enc_ref_512_16_msle_g_relu_fashion_mnist_152452
.enc_ref_512_16_msle_g_relu_fashion_mnist_152472
.enc_ref_512_16_msle_g_relu_fashion_mnist_152492
.enc_ref_512_16_msle_g_relu_fashion_mnist_152512
.enc_ref_512_16_msle_g_relu_fashion_mnist_152532
.dec_ref_512_16_msle_g_relu_fashion_mnist_152582
.dec_ref_512_16_msle_g_relu_fashion_mnist_152602
.dec_ref_512_16_msle_g_relu_fashion_mnist_152622
.dec_ref_512_16_msle_g_relu_fashion_mnist_15264
identity

identity_1Ђ@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ"ref_enc_d1/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCallЋ
@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallencoder_input.enc_ref_512_16_msle_g_relu_fashion_mnist_15243.enc_ref_512_16_msle_g_relu_fashion_mnist_15245.enc_ref_512_16_msle_g_relu_fashion_mnist_15247.enc_ref_512_16_msle_g_relu_fashion_mnist_15249.enc_ref_512_16_msle_g_relu_fashion_mnist_15251.enc_ref_512_16_msle_g_relu_fashion_mnist_15253*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *l
fgRe
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_149272B
@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallм
@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallIenc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:2.dec_ref_512_16_msle_g_relu_fashion_mnist_15258.dec_ref_512_16_msle_g_relu_fashion_mnist_15260.dec_ref_512_16_msle_g_relu_fashion_mnist_15262.dec_ref_512_16_msle_g_relu_fashion_mnist_15264*
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
GPU2*0J 8 *l
fgRe
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_150782B
@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall
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
tf_op_layer_Maximum/Maximum/yњ
tf_op_layer_Maximum/MaximumMaximumIdec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Maximum/Maximumт
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallencoder_input.enc_ref_512_16_msle_g_relu_fashion_mnist_15243.enc_ref_512_16_msle_g_relu_fashion_mnist_15245*
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
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_147252$
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
E__inference_activation_layer_call_and_return_conditional_losses_147462
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
tf_op_layer_Log_1/Log_1
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0.enc_ref_512_16_msle_g_relu_fashion_mnist_15251.enc_ref_512_16_msle_g_relu_fashion_mnist_15253*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCallї
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0.enc_ref_512_16_msle_g_relu_fashion_mnist_15247.enc_ref_512_16_msle_g_relu_fashion_mnist_15249*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
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
:џџџџџџџџџ2
tf_op_layer_AddV2_2/AddV2_2Ў
tf_op_layer_Square/SquareSquare+ref_z_mean/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Square/SquareЂ
tf_op_layer_Exp/ExpExp.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
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
:џџџџџџџџџ2
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
C__inference_add_loss_layer_call_and_return_conditional_losses_152282
add_loss/PartitionedCall
IdentityIdentityIdec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0A^dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallA^enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityр

Identity_1Identity!add_loss/PartitionedCall:output:1A^dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallA^enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:џџџџџџџџџ::::::::::2
@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall2
@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:W S
(
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameencoder_input
д
А
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_14790

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Й
c
G__inference_activation_1_layer_call_and_return_conditional_losses_14981

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
К	

H__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15405
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
identityЂStatefulPartitionedCall
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
GPU2*0J 8 *l
fgRe
c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_153812
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
и
љ
H__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15881

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2ЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *l
fgRe
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_149272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

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
в
D
(__inference_add_loss_layer_call_fn_16021

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
И%
Х
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15839

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
:	*
dtype02"
 ref_z_mean/MatMul/ReadVariableOpЋ
ref_z_mean/MatMulMatMulactivation/Relu:activations:0(ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_mean/MatMul­
!ref_z_mean/BiasAdd/ReadVariableOpReadVariableOp*ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ref_z_mean/BiasAdd/ReadVariableOp­
ref_z_mean/BiasAddBiasAddref_z_mean/MatMul:product:0)ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_mean/BiasAddИ
#ref_z_log_var/MatMul/ReadVariableOpReadVariableOp,ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#ref_z_log_var/MatMul/ReadVariableOpД
ref_z_log_var/MatMulMatMulactivation/Relu:activations:0+ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_log_var/MatMulЖ
$ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOp-ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ref_z_log_var/BiasAdd/ReadVariableOpЙ
ref_z_log_var/BiasAddBiasAddref_z_log_var/MatMul:product:0,ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ*
dtype02*
(ref_z/random_normal/RandomStandardNormalУ
ref_z/random_normal/mulMul1ref_z/random_normal/RandomStandardNormal:output:0#ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z/random_normal/mulЃ
ref_z/random_normalAddref_z/random_normal/mul:z:0!ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
	ref_z/mul^
	ref_z/ExpExpref_z/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	ref_z/Exp{
ref_z/mul_1Mulref_z/Exp:y:0ref_z/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z/mul_1
	ref_z/addAddV2ref_z_mean/BiasAdd:output:0ref_z/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	ref_z/addo
IdentityIdentityref_z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityv

Identity_1Identityref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1e

Identity_2Identityref_z/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

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
з
Г
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15032
	latent_in
decoder_d1_15020
decoder_d1_15022
decoder_output_15026
decoder_output_15028
identityЂ"decoder_d1/StatefulPartitionedCallЂ&decoder_output/StatefulPartitionedCallЂ
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCall	latent_indecoder_d1_15020decoder_d1_15022*
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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_149812
activation_1/PartitionedCallв
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_15026decoder_output_15028*
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
I__inference_decoder_output_layer_call_and_return_conditional_losses_150002(
&decoder_output/StatefulPartitionedCallв
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	latent_in
А

c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15625

inputsV
Renc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resourceW
Senc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resourceV
Renc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resourceW
Senc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resourceY
Uenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resourceZ
Venc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resourceV
Rdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_matmul_readvariableop_resourceW
Sdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_biasadd_readvariableop_resourceZ
Vdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_matmul_readvariableop_resource[
Wdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_biasadd_readvariableop_resource
identity

identity_1Ћ
Ienc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOpReadVariableOpRenc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02K
Ienc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp
:enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulMatMulinputsQenc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2<
:enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulЉ
Jenc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpSenc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02L
Jenc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpв
;enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddBiasAddDenc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul:product:0Renc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2=
;enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddѕ
8enc-ref-512-16-msle-g-relu-fashion_mnist/activation/ReluReluDenc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2:
8enc-ref-512-16-msle-g-relu-fashion_mnist/activation/ReluЊ
Ienc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpReadVariableOpRenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02K
Ienc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpЯ
:enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMulMatMulFenc-ref-512-16-msle-g-relu-fashion_mnist/activation/Relu:activations:0Qenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2<
:enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMulЈ
Jenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpSenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02L
Jenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpб
;enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAddBiasAddDenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul:product:0Renc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2=
;enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAddГ
Lenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpReadVariableOpUenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02N
Lenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpи
=enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulMatMulFenc-ref-512-16-msle-g-relu-fashion_mnist/activation/Relu:activations:0Tenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2?
=enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulБ
Menc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpVenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02O
Menc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpн
>enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddBiasAddGenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul:product:0Uenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2@
>enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddр
4enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/ShapeShapeDenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd:output:0*
T0*
_output_shapes
:26
4enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/ShapeЫ
Aenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2C
Aenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/meanЯ
Cenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2E
Cenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/stddevМ
Qenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormalRandomStandardNormal=enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02S
Qenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormalч
@enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/mulMulZenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormal:output:0Lenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2B
@enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/mulЧ
<enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normalAddDenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/mul:z:0Jenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2>
<enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normalБ
4enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?26
4enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul/xЉ
2enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mulMul=enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul/x:output:0Genc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ24
2enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mulй
2enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/ExpExp6enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ24
2enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/Exp
4enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul_1Mul6enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/Exp:y:0@enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ26
4enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul_1Ѓ
2enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/addAddV2Denc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd:output:08enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ24
2enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/addЊ
Idec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOpReadVariableOpRdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02K
Idec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOpР
:dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMulMatMul6enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/add:z:0Qdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2<
:dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMulЉ
Jdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOpReadVariableOpSdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02L
Jdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOpв
;dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAddBiasAddDdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul:product:0Rdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2=
;dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAddљ
:dec-ref-512-16-msle-g-relu-fashion_mnist/activation_1/ReluReluDdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2<
:dec-ref-512-16-msle-g-relu-fashion_mnist/activation_1/ReluЗ
Mdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOpReadVariableOpVdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02O
Mdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOpо
>dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMulMatMulHdec-ref-512-16-msle-g-relu-fashion_mnist/activation_1/Relu:activations:0Udec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2@
>dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMulЕ
Ndec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOpReadVariableOpWdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02P
Ndec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOpт
?dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAddBiasAddHdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul:product:0Vdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2A
?dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd
?dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/SigmoidSigmoidHdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2A
?dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/Sigmoid
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
tf_op_layer_Maximum/Maximum/yє
tf_op_layer_Maximum/MaximumMaximumCdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/Sigmoid:y:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Maximum/Maximumй
 ref_enc_d1/MatMul/ReadVariableOpReadVariableOpRenc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 ref_enc_d1/MatMul/ReadVariableOp
ref_enc_d1/MatMulMatMulinputs(ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ref_enc_d1/MatMulз
!ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpSenc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resource*
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
tf_op_layer_Log_1/Log_1с
#ref_z_log_var/MatMul/ReadVariableOpReadVariableOpUenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#ref_z_log_var/MatMul/ReadVariableOpД
ref_z_log_var/MatMulMatMulactivation/Relu:activations:0+ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_log_var/MatMulп
$ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpVenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ref_z_log_var/BiasAdd/ReadVariableOpЙ
ref_z_log_var/BiasAddBiasAddref_z_log_var/MatMul:product:0,ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_log_var/BiasAddи
 ref_z_mean/MatMul/ReadVariableOpReadVariableOpRenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ref_z_mean/MatMul/ReadVariableOpЋ
ref_z_mean/MatMulMatMulactivation/Relu:activations:0(ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_mean/MatMulж
!ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpSenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ref_z_mean/BiasAdd/ReadVariableOp­
ref_z_mean/BiasAddBiasAddref_z_mean/MatMul:product:0)ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
tf_op_layer_AddV2_2/AddV2_2
tf_op_layer_Square/SquareSquareref_z_mean/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Square/Square
tf_op_layer_Exp/ExpExpref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
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
:џџџџџџџџџ2
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
tf_op_layer_Mean_1/Mean_1
IdentityIdentityCdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/Sigmoid:y:0*
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
и
љ
H__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15860

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2ЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *l
fgRe
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_148832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

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
Q
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
value3B1 B+_temp_f13c774b40ec4fa5841536a09adb6d2a/part2	
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
ShardedFilenameи
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ъ
valueрBн&B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
::	::	:: : : : : :	::
:: : :
::	::	::	::
::
::	::	::	::
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
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::
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
:	:!
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
:	: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	:!
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
:	: 

_output_shapes
::% !

_output_shapes
:	: !

_output_shapes
::%"!

_output_shapes
:	:!#
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
А

c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15715

inputsV
Renc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resourceW
Senc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resourceV
Renc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resourceW
Senc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resourceY
Uenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resourceZ
Venc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resourceV
Rdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_matmul_readvariableop_resourceW
Sdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_biasadd_readvariableop_resourceZ
Vdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_matmul_readvariableop_resource[
Wdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_biasadd_readvariableop_resource
identity

identity_1Ћ
Ienc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOpReadVariableOpRenc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02K
Ienc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp
:enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulMatMulinputsQenc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2<
:enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulЉ
Jenc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpSenc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02L
Jenc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpв
;enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddBiasAddDenc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul:product:0Renc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2=
;enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddѕ
8enc-ref-512-16-msle-g-relu-fashion_mnist/activation/ReluReluDenc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2:
8enc-ref-512-16-msle-g-relu-fashion_mnist/activation/ReluЊ
Ienc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpReadVariableOpRenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02K
Ienc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpЯ
:enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMulMatMulFenc-ref-512-16-msle-g-relu-fashion_mnist/activation/Relu:activations:0Qenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2<
:enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMulЈ
Jenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpSenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02L
Jenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpб
;enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAddBiasAddDenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul:product:0Renc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2=
;enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAddГ
Lenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpReadVariableOpUenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02N
Lenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpи
=enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulMatMulFenc-ref-512-16-msle-g-relu-fashion_mnist/activation/Relu:activations:0Tenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2?
=enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulБ
Menc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpVenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02O
Menc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpн
>enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddBiasAddGenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul:product:0Uenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2@
>enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddр
4enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/ShapeShapeDenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd:output:0*
T0*
_output_shapes
:26
4enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/ShapeЫ
Aenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2C
Aenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/meanЯ
Cenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2E
Cenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/stddevМ
Qenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormalRandomStandardNormal=enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02S
Qenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormalч
@enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/mulMulZenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormal:output:0Lenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2B
@enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/mulЧ
<enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normalAddDenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/mul:z:0Jenc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2>
<enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normalБ
4enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?26
4enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul/xЉ
2enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mulMul=enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul/x:output:0Genc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ24
2enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mulй
2enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/ExpExp6enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ24
2enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/Exp
4enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul_1Mul6enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/Exp:y:0@enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ26
4enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul_1Ѓ
2enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/addAddV2Denc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd:output:08enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ24
2enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/addЊ
Idec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOpReadVariableOpRdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02K
Idec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOpР
:dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMulMatMul6enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/add:z:0Qdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2<
:dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMulЉ
Jdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOpReadVariableOpSdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02L
Jdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOpв
;dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAddBiasAddDdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul:product:0Rdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2=
;dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAddљ
:dec-ref-512-16-msle-g-relu-fashion_mnist/activation_1/ReluReluDdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2<
:dec-ref-512-16-msle-g-relu-fashion_mnist/activation_1/ReluЗ
Mdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOpReadVariableOpVdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02O
Mdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOpо
>dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMulMatMulHdec-ref-512-16-msle-g-relu-fashion_mnist/activation_1/Relu:activations:0Udec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2@
>dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMulЕ
Ndec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOpReadVariableOpWdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02P
Ndec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOpт
?dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAddBiasAddHdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul:product:0Vdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2A
?dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd
?dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/SigmoidSigmoidHdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2A
?dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/Sigmoid
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
tf_op_layer_Maximum/Maximum/yє
tf_op_layer_Maximum/MaximumMaximumCdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/Sigmoid:y:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Maximum/Maximumй
 ref_enc_d1/MatMul/ReadVariableOpReadVariableOpRenc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02"
 ref_enc_d1/MatMul/ReadVariableOp
ref_enc_d1/MatMulMatMulinputs(ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
ref_enc_d1/MatMulз
!ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpSenc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resource*
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
tf_op_layer_Log_1/Log_1с
#ref_z_log_var/MatMul/ReadVariableOpReadVariableOpUenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#ref_z_log_var/MatMul/ReadVariableOpД
ref_z_log_var/MatMulMatMulactivation/Relu:activations:0+ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_log_var/MatMulп
$ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpVenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ref_z_log_var/BiasAdd/ReadVariableOpЙ
ref_z_log_var/BiasAddBiasAddref_z_log_var/MatMul:product:0,ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_log_var/BiasAddи
 ref_z_mean/MatMul/ReadVariableOpReadVariableOpRenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 ref_z_mean/MatMul/ReadVariableOpЋ
ref_z_mean/MatMulMatMulactivation/Relu:activations:0(ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_mean/MatMulж
!ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpSenc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ref_z_mean/BiasAdd/ReadVariableOp­
ref_z_mean/BiasAddBiasAddref_z_mean/MatMul:product:0)ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
tf_op_layer_AddV2_2/AddV2_2
tf_op_layer_Square/SquareSquareref_z_mean/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Square/Square
tf_op_layer_Exp/ExpExpref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
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
:џџџџџџџџџ2
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
tf_op_layer_Mean_1/Mean_1
IdentityIdentityCdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/Sigmoid:y:0*
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
ъ

-__inference_ref_z_log_var_layer_call_fn_15991

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
:џџџџџџџџџ*$
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
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
д
А
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_15982

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Й
c
G__inference_activation_1_layer_call_and_return_conditional_losses_16067

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
Ѕ	

H__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15741

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
identityЂStatefulPartitionedCall
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
GPU2*0J 8 *l
fgRe
c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_153812
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
ж
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
Ѕ	

H__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15767

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
identityЂStatefulPartitionedCall
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
GPU2*0J 8 *l
fgRe
c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_154762
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
Ю
А
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15050

inputs
decoder_d1_15038
decoder_d1_15040
decoder_output_15044
decoder_output_15046
identityЂ"decoder_d1/StatefulPartitionedCallЂ&decoder_output/StatefulPartitionedCall
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_d1_15038decoder_d1_15040*
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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_149812
activation_1/PartitionedCallв
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_15044decoder_output_15046*
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
I__inference_decoder_output_layer_call_and_return_conditional_losses_150002(
&decoder_output/StatefulPartitionedCallв
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
г
­
E__inference_decoder_d1_layer_call_and_return_conditional_losses_14960

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
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
:џџџџџџџџџ:::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
э

H__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_14902
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2ЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *l
fgRe
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_148832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

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
у

*__inference_decoder_d1_layer_call_fn_16062

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
E__inference_decoder_d1_layer_call_and_return_conditional_losses_149602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
щ
О
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15089
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallД
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
GPU2*0J 8 *l
fgRe
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_150782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	latent_in

О
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_14883

inputs
ref_enc_d1_14863
ref_enc_d1_14865
ref_z_mean_14869
ref_z_mean_14871
ref_z_log_var_14874
ref_z_log_var_14876
identity

identity_1

identity_2Ђ"ref_enc_d1/StatefulPartitionedCallЂref_z/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCall
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallinputsref_enc_d1_14863ref_enc_d1_14865*
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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_147462
activation/PartitionedCallЛ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_14869ref_z_mean_14871*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
"ref_z_mean/StatefulPartitionedCallЪ
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_14874ref_z_log_var_14876*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCallЛ
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
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
:џџџџџџџџџ2

Identity

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity&ref_z/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

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
Є
Х
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_14834
encoder_input
ref_enc_d1_14736
ref_enc_d1_14738
ref_z_mean_14775
ref_z_mean_14777
ref_z_log_var_14801
ref_z_log_var_14803
identity

identity_1

identity_2Ђ"ref_enc_d1/StatefulPartitionedCallЂref_z/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCallІ
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputref_enc_d1_14736ref_enc_d1_14738*
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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_147462
activation/PartitionedCallЛ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_14775ref_z_mean_14777*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
"ref_z_mean/StatefulPartitionedCallЪ
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_14801ref_z_log_var_14803*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCallЛ
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
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
:џџџџџџџџџ2

Identity

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity&ref_z/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

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
Й
Б
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

ћ
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
identity_38ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9о
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*ъ
valueрBн&B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
И%
Х
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15803

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
:	*
dtype02"
 ref_z_mean/MatMul/ReadVariableOpЋ
ref_z_mean/MatMulMatMulactivation/Relu:activations:0(ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_mean/MatMul­
!ref_z_mean/BiasAdd/ReadVariableOpReadVariableOp*ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ref_z_mean/BiasAdd/ReadVariableOp­
ref_z_mean/BiasAddBiasAddref_z_mean/MatMul:product:0)ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_mean/BiasAddИ
#ref_z_log_var/MatMul/ReadVariableOpReadVariableOp,ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02%
#ref_z_log_var/MatMul/ReadVariableOpД
ref_z_log_var/MatMulMatMulactivation/Relu:activations:0+ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z_log_var/MatMulЖ
$ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOp-ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ref_z_log_var/BiasAdd/ReadVariableOpЙ
ref_z_log_var/BiasAddBiasAddref_z_log_var/MatMul:product:0,ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ*
dtype02*
(ref_z/random_normal/RandomStandardNormalУ
ref_z/random_normal/mulMul1ref_z/random_normal/RandomStandardNormal:output:0#ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z/random_normal/mulЃ
ref_z/random_normalAddref_z/random_normal/mul:z:0!ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
	ref_z/mul^
	ref_z/ExpExpref_z/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	ref_z/Exp{
ref_z/mul_1Mulref_z/Exp:y:0ref_z/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ref_z/mul_1
	ref_z/addAddV2ref_z_mean/BiasAdd:output:0ref_z/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
	ref_z/addo
IdentityIdentityref_z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityv

Identity_1Identityref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1e

Identity_2Identityref_z/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

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
К	

H__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15500
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
identityЂStatefulPartitionedCall
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
GPU2*0J 8 *l
fgRe
c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_154762
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

О
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_14927

inputs
ref_enc_d1_14907
ref_enc_d1_14909
ref_z_mean_14913
ref_z_mean_14915
ref_z_log_var_14918
ref_z_log_var_14920
identity

identity_1

identity_2Ђ"ref_enc_d1/StatefulPartitionedCallЂref_z/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCall
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallinputsref_enc_d1_14907ref_enc_d1_14909*
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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_147462
activation/PartitionedCallЛ
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_14913ref_z_mean_14915*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
"ref_z_mean/StatefulPartitionedCallЪ
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_14918ref_z_log_var_14920*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCallЛ
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
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
:џџџџџџџџџ2

Identity

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity&ref_z/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

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
э

H__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_14946
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2ЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *l
fgRe
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_149272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

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
ьV

c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15240
encoder_input2
.enc_ref_512_16_msle_g_relu_fashion_mnist_151352
.enc_ref_512_16_msle_g_relu_fashion_mnist_151372
.enc_ref_512_16_msle_g_relu_fashion_mnist_151392
.enc_ref_512_16_msle_g_relu_fashion_mnist_151412
.enc_ref_512_16_msle_g_relu_fashion_mnist_151432
.enc_ref_512_16_msle_g_relu_fashion_mnist_151452
.dec_ref_512_16_msle_g_relu_fashion_mnist_151762
.dec_ref_512_16_msle_g_relu_fashion_mnist_151782
.dec_ref_512_16_msle_g_relu_fashion_mnist_151802
.dec_ref_512_16_msle_g_relu_fashion_mnist_15182
identity

identity_1Ђ@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ"ref_enc_d1/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCallЋ
@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallencoder_input.enc_ref_512_16_msle_g_relu_fashion_mnist_15135.enc_ref_512_16_msle_g_relu_fashion_mnist_15137.enc_ref_512_16_msle_g_relu_fashion_mnist_15139.enc_ref_512_16_msle_g_relu_fashion_mnist_15141.enc_ref_512_16_msle_g_relu_fashion_mnist_15143.enc_ref_512_16_msle_g_relu_fashion_mnist_15145*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *l
fgRe
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_148832B
@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallм
@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallIenc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:2.dec_ref_512_16_msle_g_relu_fashion_mnist_15176.dec_ref_512_16_msle_g_relu_fashion_mnist_15178.dec_ref_512_16_msle_g_relu_fashion_mnist_15180.dec_ref_512_16_msle_g_relu_fashion_mnist_15182*
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
GPU2*0J 8 *l
fgRe
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_150502B
@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall
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
tf_op_layer_Maximum/Maximum/yњ
tf_op_layer_Maximum/MaximumMaximumIdec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Maximum/Maximumт
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallencoder_input.enc_ref_512_16_msle_g_relu_fashion_mnist_15135.enc_ref_512_16_msle_g_relu_fashion_mnist_15137*
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
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_147252$
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
E__inference_activation_layer_call_and_return_conditional_losses_147462
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
tf_op_layer_Log_1/Log_1
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0.enc_ref_512_16_msle_g_relu_fashion_mnist_15143.enc_ref_512_16_msle_g_relu_fashion_mnist_15145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCallї
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0.enc_ref_512_16_msle_g_relu_fashion_mnist_15139.enc_ref_512_16_msle_g_relu_fashion_mnist_15141*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
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
:џџџџџџџџџ2
tf_op_layer_AddV2_2/AddV2_2Ў
tf_op_layer_Square/SquareSquare+ref_z_mean/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Square/SquareЂ
tf_op_layer_Exp/ExpExp.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
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
:џџџџџџџџџ2
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
C__inference_add_loss_layer_call_and_return_conditional_losses_152282
add_loss/PartitionedCall
IdentityIdentityIdec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0A^dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallA^enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityр

Identity_1Identity!add_loss/PartitionedCall:output:1A^dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallA^enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:џџџџџџџџџ::::::::::2
@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall2
@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:W S
(
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameencoder_input
зф
ы

 __inference__wrapped_model_14711
encoder_input
{vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resource
|vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resource
{vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resource
|vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resource
~vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resource
vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resource
{vae_ref_512_16_msle_g_relu_fashion_mnist_dec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_matmul_readvariableop_resource
|vae_ref_512_16_msle_g_relu_fashion_mnist_dec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_biasadd_readvariableop_resource
vae_ref_512_16_msle_g_relu_fashion_mnist_dec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_matmul_readvariableop_resource
vae_ref_512_16_msle_g_relu_fashion_mnist_dec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_biasadd_readvariableop_resource
identityІ
rvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOpReadVariableOp{vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02t
rvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp
cvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulMatMulencoder_inputzvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2e
cvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulЄ
svae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOp|vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02u
svae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpі
dvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddBiasAddmvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul:product:0{vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2f
dvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd№
avae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/activation/ReluRelumvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2c
avae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/activation/ReluЅ
rvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpReadVariableOp{vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02t
rvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpѓ
cvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMulMatMulovae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/activation/Relu:activations:0zvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2e
cvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMulЃ
svae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpReadVariableOp|vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02u
svae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpѕ
dvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAddBiasAddmvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul:product:0{vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2f
dvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAddЎ
uvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpReadVariableOp~vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02w
uvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpќ
fvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulMatMulovae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/activation/Relu:activations:0}vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2h
fvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulЌ
vvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpvae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02x
vvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOp
gvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddBiasAddpvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul:product:0~vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2i
gvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddл
]vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/ShapeShapemvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2_
]vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/Shape
jvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2l
jvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/meanЁ
lvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2n
lvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/stddevЗ
zvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormalRandomStandardNormalfvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02|
zvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormal
ivae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/mulMulvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/RandomStandardNormal:output:0uvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2k
ivae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/mulы
evae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normalAddmvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/mul:z:0svae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2g
evae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal
]vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2_
]vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul/xЭ
[vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mulMulfvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul/x:output:0pvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2]
[vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mulд
[vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/ExpExp_vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2]
[vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/ExpУ
]vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul_1Mul_vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/Exp:y:0ivae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2_
]vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul_1Ч
[vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/addAddV2mvae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd:output:0avae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2]
[vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/addЅ
rvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOpReadVariableOp{vae_ref_512_16_msle_g_relu_fashion_mnist_dec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02t
rvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOpф
cvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMulMatMul_vae-ref-512-16-msle-g-relu-fashion_mnist/enc-ref-512-16-msle-g-relu-fashion_mnist/ref_z/add:z:0zvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2e
cvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMulЄ
svae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOpReadVariableOp|vae_ref_512_16_msle_g_relu_fashion_mnist_dec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02u
svae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOpі
dvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAddBiasAddmvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul:product:0{vae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2f
dvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAddє
cvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/activation_1/ReluRelumvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2e
cvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/activation_1/ReluВ
vvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOpReadVariableOpvae_ref_512_16_msle_g_relu_fashion_mnist_dec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02x
vvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOp
gvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMulMatMulqvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/activation_1/Relu:activations:0~vae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2i
gvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMulБ
wvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOpReadVariableOpvae_ref_512_16_msle_g_relu_fashion_mnist_dec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02y
wvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOp
hvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAddBiasAddqvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul:product:0vae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd
hvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/SigmoidSigmoidqvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2j
hvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/Sigmoidн
Jvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32L
Jvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Maximum_1/Maximum_1/yХ
Hvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Maximum_1/Maximum_1Maximumencoder_inputSvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2J
Hvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Maximum_1/Maximum_1е
Fvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж32H
Fvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Maximum/Maximum/y
Dvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Maximum/MaximumMaximumlvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/Sigmoid:y:0Ovae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2F
Dvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Maximum/Maximumд
Ivae-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOpReadVariableOp{vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02K
Ivae-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp
:vae-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulMatMulencoder_inputQvae-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2<
:vae-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMulв
Jvae-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOp|vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02L
Jvae-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOpв
;vae-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddBiasAddDvae-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/MatMul:product:0Rvae-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2=
;vae-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAddе
Fvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_1/AddV2_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2H
Fvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_1/AddV2_1/yі
Dvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_1/AddV2_1AddV2Lvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Maximum_1/Maximum_1:z:0Ovae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_1/AddV2_1/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2F
Dvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_1/AddV2_1Э
Bvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2/AddV2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2D
Bvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2/AddV2/yц
@vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2/AddV2AddV2Hvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Maximum/Maximum:z:0Kvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2/AddV2/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2B
@vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2/AddV2ѕ
8vae-ref-512-16-msle-g-relu-fashion_mnist/activation/ReluReluDvae-ref-512-16-msle-g-relu-fashion_mnist/ref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2:
8vae-ref-512-16-msle-g-relu-fashion_mnist/activation/Relu
<vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Log/LogLogDvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2/AddV2:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2>
<vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Log/Log
@vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Log_1/Log_1LogHvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_1/AddV2_1:z:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2B
@vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Log_1/Log_1м
Lvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpReadVariableOp~vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02N
Lvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOpи
=vae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulMatMulFvae-ref-512-16-msle-g-relu-fashion_mnist/activation/Relu:activations:0Tvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2?
=vae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMulк
Mvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpvae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02O
Mvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOpн
>vae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddBiasAddGvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/MatMul:product:0Uvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2@
>vae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAddг
Ivae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpReadVariableOp{vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02K
Ivae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOpЯ
:vae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMulMatMulFvae-ref-512-16-msle-g-relu-fashion_mnist/activation/Relu:activations:0Qvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2<
:vae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMulб
Jvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpReadVariableOp|vae_ref_512_16_msle_g_relu_fashion_mnist_enc_ref_512_16_msle_g_relu_fashion_mnist_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02L
Jvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOpб
;vae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAddBiasAddDvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/MatMul:product:0Rvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2=
;vae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd
Xvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_SquaredDifference/SquaredDifferenceSquaredDifference@vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Log/Log:y:0Dvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Log_1/Log_1:y:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2Z
Xvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_SquaredDifference/SquaredDifferenceе
Fvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_2/AddV2_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2H
Fvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_2/AddV2_2/x№
Dvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_2/AddV2_2AddV2Ovae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_2/AddV2_2/x:output:0Gvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2F
Dvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_2/AddV2_2
Bvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Square/SquareSquareDvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_mean/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2D
Bvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Square/Square
<vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Exp/ExpExpGvae-ref-512-16-msle-g-relu-fashion_mnist/ref_z_log_var/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2>
<vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Exp/Expя
Pvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2R
Pvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mean/Mean/reduction_indicesў
>vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mean/MeanMean\vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_SquaredDifference/SquaredDifference:z:0Yvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mean/Mean/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2@
>vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mean/Meanж
<vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Sub/SubSubHvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_2/AddV2_2:z:0Fvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Square/Square:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2>
<vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Sub/SubХ
>vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mul/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  DD2@
>vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mul/Mul/yв
<vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mul/MulMulGvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mean/Mean:output:0Gvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mul/Mul/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2>
<vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mul/Mulа
@vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Sub_1/Sub_1Sub@vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Sub/Sub:z:0@vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Exp/Exp:y:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2B
@vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Sub_1/Sub_1ы
Nvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2P
Nvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Sum/Sum/reduction_indicesп
<vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Sum/SumSumDvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Sub_1/Sub_1:z:0Wvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2>
<vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Sum/SumЭ
Bvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mul_1/Mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   П2D
Bvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mul_1/Mul_1/yм
@vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mul_1/Mul_1MulEvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Sum/Sum:output:0Kvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mul_1/Mul_1/y:output:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2B
@vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mul_1/Mul_1к
Dvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_3/AddV2_3AddV2@vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mul/Mul:z:0Dvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mul_1/Mul_1:z:0*
T0*
_cloned(*#
_output_shapes
:џџџџџџџџџ2F
Dvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_3/AddV2_3і
Tvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mean_1/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2V
Tvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mean_1/Mean_1/reduction_indicesщ
Bvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mean_1/Mean_1MeanHvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_AddV2_3/AddV2_3:z:0]vae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mean_1/Mean_1/reduction_indices:output:0*
T0*
_cloned(*
_output_shapes
: 2D
Bvae-ref-512-16-msle-g-relu-fashion_mnist/tf_op_layer_Mean_1/Mean_1С
IdentityIdentitylvae-ref-512-16-msle-g-relu-fashion_mnist/dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/Sigmoid:y:0*
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
E__inference_activation_layer_call_and_return_conditional_losses_15967

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
ЩV

c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15476

inputs2
.enc_ref_512_16_msle_g_relu_fashion_mnist_154102
.enc_ref_512_16_msle_g_relu_fashion_mnist_154122
.enc_ref_512_16_msle_g_relu_fashion_mnist_154142
.enc_ref_512_16_msle_g_relu_fashion_mnist_154162
.enc_ref_512_16_msle_g_relu_fashion_mnist_154182
.enc_ref_512_16_msle_g_relu_fashion_mnist_154202
.dec_ref_512_16_msle_g_relu_fashion_mnist_154252
.dec_ref_512_16_msle_g_relu_fashion_mnist_154272
.dec_ref_512_16_msle_g_relu_fashion_mnist_154292
.dec_ref_512_16_msle_g_relu_fashion_mnist_15431
identity

identity_1Ђ@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ"ref_enc_d1/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCallЄ
@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallinputs.enc_ref_512_16_msle_g_relu_fashion_mnist_15410.enc_ref_512_16_msle_g_relu_fashion_mnist_15412.enc_ref_512_16_msle_g_relu_fashion_mnist_15414.enc_ref_512_16_msle_g_relu_fashion_mnist_15416.enc_ref_512_16_msle_g_relu_fashion_mnist_15418.enc_ref_512_16_msle_g_relu_fashion_mnist_15420*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *l
fgRe
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_149272B
@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallм
@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallIenc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:2.dec_ref_512_16_msle_g_relu_fashion_mnist_15425.dec_ref_512_16_msle_g_relu_fashion_mnist_15427.dec_ref_512_16_msle_g_relu_fashion_mnist_15429.dec_ref_512_16_msle_g_relu_fashion_mnist_15431*
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
GPU2*0J 8 *l
fgRe
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_150782B
@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall
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
tf_op_layer_Maximum/Maximum/yњ
tf_op_layer_Maximum/MaximumMaximumIdec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Maximum/Maximumл
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallinputs.enc_ref_512_16_msle_g_relu_fashion_mnist_15410.enc_ref_512_16_msle_g_relu_fashion_mnist_15412*
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
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_147252$
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
E__inference_activation_layer_call_and_return_conditional_losses_147462
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
tf_op_layer_Log_1/Log_1
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0.enc_ref_512_16_msle_g_relu_fashion_mnist_15418.enc_ref_512_16_msle_g_relu_fashion_mnist_15420*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCallї
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0.enc_ref_512_16_msle_g_relu_fashion_mnist_15414.enc_ref_512_16_msle_g_relu_fashion_mnist_15416*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
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
:џџџџџџџџџ2
tf_op_layer_AddV2_2/AddV2_2Ў
tf_op_layer_Square/SquareSquare+ref_z_mean/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Square/SquareЂ
tf_op_layer_Exp/ExpExp.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
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
:џџџџџџџџџ2
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
C__inference_add_loss_layer_call_and_return_conditional_losses_152282
add_loss/PartitionedCall
IdentityIdentityIdec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0A^dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallA^enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityр

Identity_1Identity!add_loss/PartitionedCall:output:1A^dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallA^enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:џџџџџџџџџ::::::::::2
@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall2
@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЩV

c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15381

inputs2
.enc_ref_512_16_msle_g_relu_fashion_mnist_153152
.enc_ref_512_16_msle_g_relu_fashion_mnist_153172
.enc_ref_512_16_msle_g_relu_fashion_mnist_153192
.enc_ref_512_16_msle_g_relu_fashion_mnist_153212
.enc_ref_512_16_msle_g_relu_fashion_mnist_153232
.enc_ref_512_16_msle_g_relu_fashion_mnist_153252
.dec_ref_512_16_msle_g_relu_fashion_mnist_153302
.dec_ref_512_16_msle_g_relu_fashion_mnist_153322
.dec_ref_512_16_msle_g_relu_fashion_mnist_153342
.dec_ref_512_16_msle_g_relu_fashion_mnist_15336
identity

identity_1Ђ@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallЂ"ref_enc_d1/StatefulPartitionedCallЂ%ref_z_log_var/StatefulPartitionedCallЂ"ref_z_mean/StatefulPartitionedCallЄ
@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallinputs.enc_ref_512_16_msle_g_relu_fashion_mnist_15315.enc_ref_512_16_msle_g_relu_fashion_mnist_15317.enc_ref_512_16_msle_g_relu_fashion_mnist_15319.enc_ref_512_16_msle_g_relu_fashion_mnist_15321.enc_ref_512_16_msle_g_relu_fashion_mnist_15323.enc_ref_512_16_msle_g_relu_fashion_mnist_15325*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *l
fgRe
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_148832B
@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallм
@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallStatefulPartitionedCallIenc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:2.dec_ref_512_16_msle_g_relu_fashion_mnist_15330.dec_ref_512_16_msle_g_relu_fashion_mnist_15332.dec_ref_512_16_msle_g_relu_fashion_mnist_15334.dec_ref_512_16_msle_g_relu_fashion_mnist_15336*
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
GPU2*0J 8 *l
fgRe
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_150502B
@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall
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
tf_op_layer_Maximum/Maximum/yњ
tf_op_layer_Maximum/MaximumMaximumIdec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*(
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Maximum/Maximumл
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallinputs.enc_ref_512_16_msle_g_relu_fashion_mnist_15315.enc_ref_512_16_msle_g_relu_fashion_mnist_15317*
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
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_147252$
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
E__inference_activation_layer_call_and_return_conditional_losses_147462
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
tf_op_layer_Log_1/Log_1
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0.enc_ref_512_16_msle_g_relu_fashion_mnist_15323.enc_ref_512_16_msle_g_relu_fashion_mnist_15325*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_147902'
%ref_z_log_var/StatefulPartitionedCallї
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0.enc_ref_512_16_msle_g_relu_fashion_mnist_15319.enc_ref_512_16_msle_g_relu_fashion_mnist_15321*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_147642$
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
:џџџџџџџџџ2
tf_op_layer_AddV2_2/AddV2_2Ў
tf_op_layer_Square/SquareSquare+ref_z_mean/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
tf_op_layer_Square/SquareЂ
tf_op_layer_Exp/ExpExp.ref_z_log_var/StatefulPartitionedCall:output:0*
T0*
_cloned(*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
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
:џџџџџџџџџ2
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
C__inference_add_loss_layer_call_and_return_conditional_losses_152282
add_loss/PartitionedCall
IdentityIdentityIdec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall:output:0A^dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallA^enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityр

Identity_1Identity!add_loss/PartitionedCall:output:1A^dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCallA^enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall#^ref_enc_d1/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*O
_input_shapes>
<:џџџџџџџџџ::::::::::2
@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall@dec-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall2
@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall@enc-ref-512-16-msle-g-relu-fashion_mnist/StatefulPartitionedCall2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ю
А
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15078

inputs
decoder_d1_15066
decoder_d1_15068
decoder_output_15072
decoder_output_15074
identityЂ"decoder_d1/StatefulPartitionedCallЂ&decoder_output/StatefulPartitionedCall
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_d1_15066decoder_d1_15068*
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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_149812
activation_1/PartitionedCallв
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_15072decoder_output_15074*
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
I__inference_decoder_output_layer_call_and_return_conditional_losses_150002(
&decoder_output/StatefulPartitionedCallв
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ж
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
р
Л
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15930

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallБ
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
GPU2*0J 8 *l
fgRe
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_150502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

F
*__inference_activation_layer_call_fn_15972

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
E__inference_activation_layer_call_and_return_conditional_losses_147462
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

n
%__inference_ref_z_layer_call_fn_16043
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
:џџџџџџџџџ* 
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
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
З
a
E__inference_activation_layer_call_and_return_conditional_losses_14746

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
Љ
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
random_normal/stddevЏ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02$
"random_normal/RandomStandardNormalЋ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1

Ш
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15917

inputs-
)decoder_d1_matmul_readvariableop_resource.
*decoder_d1_biasadd_readvariableop_resource1
-decoder_output_matmul_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identityЏ
 decoder_d1/MatMul/ReadVariableOpReadVariableOp)decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
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
#:џџџџџџџџџ:::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ю

.__inference_decoder_output_layer_call_fn_16092

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
I__inference_decoder_output_layer_call_and_return_conditional_losses_150002
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
у

*__inference_ref_z_mean_layer_call_fn_16010

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
:џџџџџџџџџ*$
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
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ш
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15899

inputs-
)decoder_d1_matmul_readvariableop_resource.
*decoder_d1_biasadd_readvariableop_resource1
-decoder_output_matmul_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identityЏ
 decoder_d1/MatMul/ReadVariableOpReadVariableOp)decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
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
#:џџџџџџџџџ:::::O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
х

*__inference_ref_enc_d1_layer_call_fn_15962

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
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_147252
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
з
Г
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15017
	latent_in
decoder_d1_14971
decoder_d1_14973
decoder_output_15011
decoder_output_15013
identityЂ"decoder_d1/StatefulPartitionedCallЂ&decoder_output/StatefulPartitionedCallЂ
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCall	latent_indecoder_d1_14971decoder_d1_14973*
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
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_149812
activation_1/PartitionedCallв
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_15011decoder_output_15013*
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
I__inference_decoder_output_layer_call_and_return_conditional_losses_150002(
&decoder_output/StatefulPartitionedCallв
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	latent_in
б
­
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_14764

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
щ
О
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15061
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallД
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
GPU2*0J 8 *l
fgRe
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_150502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	latent_in
Я
ї
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
 __inference__wrapped_model_147112
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
Й
Б
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
random_normal/stddevЏ
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype02$
"random_normal/RandomStandardNormalЋ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
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
:џџџџџџџџџ2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
б
­
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_16001

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є
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

H
,__inference_activation_1_layer_call_fn_16072

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
G__inference_activation_1_layer_call_and_return_conditional_losses_149812
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
 
_user_specified_nameinputs"ФL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*й
serving_defaultХ
H
encoder_input7
serving_default_encoder_input:0џџџџџџџџџ]
(dec-ref-512-16-msle-g-relu-fashion_mnist1
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ль
ьд
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
trainable_variables
regularization_losses
	variables
 	keras_api
!
signatures
Щ__call__
+Ъ&call_and_return_all_conditional_losses
Ы_default_save_signature"ЏЯ
_tf_keras_networkЯ{"class_name": "Functional", "name": "vae-ref-512-16-msle-g-relu-fashion_mnist", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "vae-ref-512-16-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "enc-ref-512-16-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}, "name": "ref_z", "inbound_nodes": [[["ref_z_mean", 0, 0, {}], ["ref_z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["ref_z_mean", 0, 0], ["ref_z_log_var", 0, 0], ["ref_z", 0, 0]]}, "name": "enc-ref-512-16-msle-g-relu-fashion_mnist", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "dec-ref-512-16-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "name": "dec-ref-512-16-msle-g-relu-fashion_mnist", "inbound_nodes": [[["enc-ref-512-16-msle-g-relu-fashion_mnist", 1, 2, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/Sigmoid", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}, "name": "tf_op_layer_Maximum", "inbound_nodes": [[["dec-ref-512-16-msle-g-relu-fashion_mnist", 1, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum_1", "op": "Maximum", "input": ["encoder_input", "Maximum_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}, "name": "tf_op_layer_Maximum_1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2", "op": "AddV2", "input": ["Maximum", "AddV2/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_AddV2", "inbound_nodes": [[["tf_op_layer_Maximum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_1", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_1", "op": "AddV2", "input": ["Maximum_1", "AddV2_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_AddV2_1", "inbound_nodes": [[["tf_op_layer_Maximum_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_2", "op": "AddV2", "input": ["AddV2_2/x", "ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 1.0}}, "name": "tf_op_layer_AddV2_2", "inbound_nodes": [[["ref_z_log_var", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Square", "trainable": true, "dtype": "float32", "node_def": {"name": "Square", "op": "Square", "input": ["ref_z_mean/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Square", "inbound_nodes": [[["ref_z_mean", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Log", "trainable": true, "dtype": "float32", "node_def": {"name": "Log", "op": "Log", "input": ["AddV2"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Log", "inbound_nodes": [[["tf_op_layer_AddV2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Log_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Log_1", "op": "Log", "input": ["AddV2_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Log_1", "inbound_nodes": [[["tf_op_layer_AddV2_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["AddV2_2", "Square"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub", "inbound_nodes": [[["tf_op_layer_AddV2_2", 0, 0, {}], ["tf_op_layer_Square", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Exp", "trainable": true, "dtype": "float32", "node_def": {"name": "Exp", "op": "Exp", "input": ["ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Exp", "inbound_nodes": [[["ref_z_log_var", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "SquaredDifference", "trainable": true, "dtype": "float32", "node_def": {"name": "SquaredDifference", "op": "SquaredDifference", "input": ["Log", "Log_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_SquaredDifference", "inbound_nodes": [[["tf_op_layer_Log", 0, 0, {}], ["tf_op_layer_Log_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub_1", "op": "Sub", "input": ["Sub", "Exp"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub_1", "inbound_nodes": [[["tf_op_layer_Sub", 0, 0, {}], ["tf_op_layer_Exp", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean", "op": "Mean", "input": ["SquaredDifference", "Mean/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Mean", "inbound_nodes": [[["tf_op_layer_SquaredDifference", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["Sub_1", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Sum", "inbound_nodes": [[["tf_op_layer_Sub_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["Mean", "Mul/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 784.0}}, "name": "tf_op_layer_Mul", "inbound_nodes": [[["tf_op_layer_Mean", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["Sum", "Mul_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": -0.5}}, "name": "tf_op_layer_Mul_1", "inbound_nodes": [[["tf_op_layer_Sum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_3", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_3", "op": "AddV2", "input": ["Mul", "Mul_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_3", "inbound_nodes": [[["tf_op_layer_Mul", 0, 0, {}], ["tf_op_layer_Mul_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean_1", "op": "Mean", "input": ["AddV2_3", "Mean_1/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Mean_1", "inbound_nodes": [[["tf_op_layer_AddV2_3", 0, 0, {}]]]}, {"class_name": "AddLoss", "config": {"name": "add_loss", "trainable": true, "dtype": "float32", "unconditional": false}, "name": "add_loss", "inbound_nodes": [[["tf_op_layer_Mean_1", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["dec-ref-512-16-msle-g-relu-fashion_mnist", 1, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "vae-ref-512-16-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "enc-ref-512-16-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}, "name": "ref_z", "inbound_nodes": [[["ref_z_mean", 0, 0, {}], ["ref_z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["ref_z_mean", 0, 0], ["ref_z_log_var", 0, 0], ["ref_z", 0, 0]]}, "name": "enc-ref-512-16-msle-g-relu-fashion_mnist", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "dec-ref-512-16-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "name": "dec-ref-512-16-msle-g-relu-fashion_mnist", "inbound_nodes": [[["enc-ref-512-16-msle-g-relu-fashion_mnist", 1, 2, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/Sigmoid", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}, "name": "tf_op_layer_Maximum", "inbound_nodes": [[["dec-ref-512-16-msle-g-relu-fashion_mnist", 1, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum_1", "op": "Maximum", "input": ["encoder_input", "Maximum_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}, "name": "tf_op_layer_Maximum_1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2", "op": "AddV2", "input": ["Maximum", "AddV2/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_AddV2", "inbound_nodes": [[["tf_op_layer_Maximum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_1", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_1", "op": "AddV2", "input": ["Maximum_1", "AddV2_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_AddV2_1", "inbound_nodes": [[["tf_op_layer_Maximum_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_2", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_2", "op": "AddV2", "input": ["AddV2_2/x", "ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 1.0}}, "name": "tf_op_layer_AddV2_2", "inbound_nodes": [[["ref_z_log_var", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Square", "trainable": true, "dtype": "float32", "node_def": {"name": "Square", "op": "Square", "input": ["ref_z_mean/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Square", "inbound_nodes": [[["ref_z_mean", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Log", "trainable": true, "dtype": "float32", "node_def": {"name": "Log", "op": "Log", "input": ["AddV2"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Log", "inbound_nodes": [[["tf_op_layer_AddV2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Log_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Log_1", "op": "Log", "input": ["AddV2_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Log_1", "inbound_nodes": [[["tf_op_layer_AddV2_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["AddV2_2", "Square"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub", "inbound_nodes": [[["tf_op_layer_AddV2_2", 0, 0, {}], ["tf_op_layer_Square", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Exp", "trainable": true, "dtype": "float32", "node_def": {"name": "Exp", "op": "Exp", "input": ["ref_z_log_var/BiasAdd"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Exp", "inbound_nodes": [[["ref_z_log_var", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "SquaredDifference", "trainable": true, "dtype": "float32", "node_def": {"name": "SquaredDifference", "op": "SquaredDifference", "input": ["Log", "Log_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_SquaredDifference", "inbound_nodes": [[["tf_op_layer_Log", 0, 0, {}], ["tf_op_layer_Log_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub_1", "op": "Sub", "input": ["Sub", "Exp"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub_1", "inbound_nodes": [[["tf_op_layer_Sub", 0, 0, {}], ["tf_op_layer_Exp", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean", "op": "Mean", "input": ["SquaredDifference", "Mean/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Mean", "inbound_nodes": [[["tf_op_layer_SquaredDifference", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["Sub_1", "Sum/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Sum", "inbound_nodes": [[["tf_op_layer_Sub_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul", "op": "Mul", "input": ["Mean", "Mul/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 784.0}}, "name": "tf_op_layer_Mul", "inbound_nodes": [[["tf_op_layer_Mean", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_1", "op": "Mul", "input": ["Sum", "Mul_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": -0.5}}, "name": "tf_op_layer_Mul_1", "inbound_nodes": [[["tf_op_layer_Sum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_3", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_3", "op": "AddV2", "input": ["Mul", "Mul_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_3", "inbound_nodes": [[["tf_op_layer_Mul", 0, 0, {}], ["tf_op_layer_Mul_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean_1", "op": "Mean", "input": ["AddV2_3", "Mean_1/reduction_indices"], "attr": {"keep_dims": {"b": false}, "Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0]}}, "name": "tf_op_layer_Mean_1", "inbound_nodes": [[["tf_op_layer_AddV2_3", 0, 0, {}]]]}, {"class_name": "AddLoss", "config": {"name": "add_loss", "trainable": true, "dtype": "float32", "unconditional": false}, "name": "add_loss", "inbound_nodes": [[["tf_op_layer_Mean_1", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["dec-ref-512-16-msle-g-relu-fashion_mnist", 1, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
љ"і
_tf_keras_input_layerж{"class_name": "InputLayer", "name": "encoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}}
*
layer-0
layer_with_weights-0
layer-1
layer-2
	layer_with_weights-1
	layer-3
layer_with_weights-2
layer-4
"layer-5
#trainable_variables
$regularization_losses
%	variables
&	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses"ф'
_tf_keras_networkШ'{"class_name": "Functional", "name": "enc-ref-512-16-msle-g-relu-fashion_mnist", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "enc-ref-512-16-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}, "name": "ref_z", "inbound_nodes": [[["ref_z_mean", 0, 0, {}], ["ref_z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["ref_z_mean", 0, 0], ["ref_z_log_var", 0, 0], ["ref_z", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "enc-ref-512-16-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}, "name": "ref_z", "inbound_nodes": [[["ref_z_mean", 0, 0, {}], ["ref_z_log_var", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["ref_z_mean", 0, 0], ["ref_z_log_var", 0, 0], ["ref_z", 0, 0]]}}}
Є
'layer-0
(layer_with_weights-0
(layer-1
)layer-2
*layer_with_weights-1
*layer-3
+trainable_variables
,regularization_losses
-	variables
.	keras_api
Ю__call__
+Я&call_and_return_all_conditional_losses"Ћ
_tf_keras_network{"class_name": "Functional", "name": "dec-ref-512-16-msle-g-relu-fashion_mnist", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "dec-ref-512-16-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "dec-ref-512-16-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}}}
§

/kernel
0bias
1trainable_variables
2regularization_losses
3	variables
4	keras_api
а__call__
+б&call_and_return_all_conditional_losses"ж
_tf_keras_layerМ{"class_name": "Dense", "name": "ref_enc_d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
г
5trainable_variables
6regularization_losses
7	variables
8	keras_api
в__call__
+г&call_and_return_all_conditional_losses"Т
_tf_keras_layerЈ{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
Ј
9	keras_api"
_tf_keras_layerќ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Maximum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/Sigmoid", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}}
ў
:	keras_api"ь
_tf_keras_layerв{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Maximum_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "Maximum_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum_1", "op": "Maximum", "input": ["encoder_input", "Maximum_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}}


;kernel
<bias
=trainable_variables
>regularization_losses
?	variables
@	keras_api
д__call__
+е&call_and_return_all_conditional_losses"л
_tf_keras_layerС{"class_name": "Dense", "name": "ref_z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
ќ

Akernel
Bbias
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"е
_tf_keras_layerЛ{"class_name": "Dense", "name": "ref_z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
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
Wtrainable_variables
Xregularization_losses
Y	variables
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

elayers
trainable_variables
flayer_metrics
gnon_trainable_variables
regularization_losses
hmetrics
	variables
Щ__call__
Ы_default_save_signature
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
-
кserving_default"
signature_map
Б
itrainable_variables
jregularization_losses
k	variables
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

nlayers
#trainable_variables
olayer_metrics
pnon_trainable_variables
$regularization_losses
qmetrics
%	variables
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
я"ь
_tf_keras_input_layerЬ{"class_name": "InputLayer", "name": "latent_in", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}}
ћ

`kernel
abias
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
н__call__
+о&call_and_return_all_conditional_losses"д
_tf_keras_layerК{"class_name": "Dense", "name": "decoder_d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
з
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
п__call__
+р&call_and_return_all_conditional_losses"Ц
_tf_keras_layerЌ{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}


bkernel
cbias
ztrainable_variables
{regularization_losses
|	variables
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

layers
+trainable_variables
layer_metrics
non_trainable_variables
,regularization_losses
metrics
-	variables
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
layers
layer_metrics
1trainable_variables
non_trainable_variables
2regularization_losses
metrics
3	variables
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
layers
layer_metrics
5trainable_variables
non_trainable_variables
6regularization_losses
metrics
7	variables
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
':%	2ref_z_log_var/kernel
 :2ref_z_log_var/bias
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
layers
layer_metrics
=trainable_variables
non_trainable_variables
>regularization_losses
metrics
?	variables
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
$:"	2ref_z_mean/kernel
:2ref_z_mean/bias
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
layers
layer_metrics
Ctrainable_variables
non_trainable_variables
Dregularization_losses
metrics
E	variables
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
layers
layer_metrics
Wtrainable_variables
non_trainable_variables
Xregularization_losses
metrics
Y	variables
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
$:"	2decoder_d1/kernel
:2decoder_d1/bias
):'
2decoder_output/kernel
": 2decoder_output/bias
 "
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
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 layer_regularization_losses
layers
layer_metrics
itrainable_variables
 non_trainable_variables
jregularization_losses
Ёmetrics
k	variables
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
Ѓlayers
Єlayer_metrics
rtrainable_variables
Ѕnon_trainable_variables
sregularization_losses
Іmetrics
t	variables
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
Јlayers
Љlayer_metrics
vtrainable_variables
Њnon_trainable_variables
wregularization_losses
Ћmetrics
x	variables
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
­layers
Ўlayer_metrics
ztrainable_variables
Џnon_trainable_variables
{regularization_losses
Аmetrics
|	variables
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
Б0
В1"
trackable_list_wrapper
.
Г	variables"
_generic_user_object
*:(
2Adam/ref_enc_d1/kernel/m
#:!2Adam/ref_enc_d1/bias/m
,:*	2Adam/ref_z_log_var/kernel/m
%:#2Adam/ref_z_log_var/bias/m
):'	2Adam/ref_z_mean/kernel/m
": 2Adam/ref_z_mean/bias/m
):'	2Adam/decoder_d1/kernel/m
#:!2Adam/decoder_d1/bias/m
.:,
2Adam/decoder_output/kernel/m
':%2Adam/decoder_output/bias/m
*:(
2Adam/ref_enc_d1/kernel/v
#:!2Adam/ref_enc_d1/bias/v
,:*	2Adam/ref_z_log_var/kernel/v
%:#2Adam/ref_z_log_var/bias/v
):'	2Adam/ref_z_mean/kernel/v
": 2Adam/ref_z_mean/bias/v
):'	2Adam/decoder_d1/kernel/v
#:!2Adam/decoder_d1/bias/v
.:,
2Adam/decoder_output/kernel/v
':%2Adam/decoder_output/bias/v
ю2ы
H__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15405
H__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15767
H__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15500
H__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15741Р
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
к2з
c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15625
c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15240
c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15715
c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15309Р
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
 __inference__wrapped_model_14711Н
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
ю2ы
H__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_14902
H__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_14946
H__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15881
H__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15860Р
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
к2з
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_14857
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_14834
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15839
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15803Р
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
ю2ы
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15930
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15061
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15089
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15943Р
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
к2з
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15017
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15917
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15899
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15032Р
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
*__inference_ref_enc_d1_layer_call_fn_15962Ђ
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
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_15953Ђ
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
*__inference_activation_layer_call_fn_15972Ђ
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
E__inference_activation_layer_call_and_return_conditional_losses_15967Ђ
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
-__inference_ref_z_log_var_layer_call_fn_15991Ђ
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
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_15982Ђ
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
*__inference_ref_z_mean_layer_call_fn_16010Ђ
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
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_16001Ђ
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
(__inference_add_loss_layer_call_fn_16021Ђ
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
C__inference_add_loss_layer_call_and_return_conditional_losses_16015Ђ
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
#__inference_signature_wrapper_15535encoder_input
Я2Ь
%__inference_ref_z_layer_call_fn_16043Ђ
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
@__inference_ref_z_layer_call_and_return_conditional_losses_16037Ђ
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
*__inference_decoder_d1_layer_call_fn_16062Ђ
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
E__inference_decoder_d1_layer_call_and_return_conditional_losses_16053Ђ
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
,__inference_activation_1_layer_call_fn_16072Ђ
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
G__inference_activation_1_layer_call_and_return_conditional_losses_16067Ђ
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
.__inference_decoder_output_layer_call_fn_16092Ђ
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
I__inference_decoder_output_layer_call_and_return_conditional_losses_16083Ђ
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
 р
 __inference__wrapped_model_14711Л
/0AB;<`abc7Ђ4
-Ђ*
(%
encoder_inputџџџџџџџџџ
Њ "tЊq
o
(dec-ref-512-16-msle-g-relu-fashion_mnistC@
(dec-ref-512-16-msle-g-relu-fashion_mnistџџџџџџџџџЅ
G__inference_activation_1_layer_call_and_return_conditional_losses_16067Z0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 }
,__inference_activation_1_layer_call_fn_16072M0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЃ
E__inference_activation_layer_call_and_return_conditional_losses_15967Z0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 {
*__inference_activation_layer_call_fn_15972M0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ
C__inference_add_loss_layer_call_and_return_conditional_losses_16015DЂ
Ђ

inputs 
Њ ""Ђ


0 

	
1/0 U
(__inference_add_loss_layer_call_fn_16021)Ђ
Ђ

inputs 
Њ " б
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15017j`abc:Ђ7
0Ђ-
# 
	latent_inџџџџџџџџџ
p

 
Њ "&Ђ#

0џџџџџџџџџ
 б
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15032j`abc:Ђ7
0Ђ-
# 
	latent_inџџџџџџџџџ
p 

 
Њ "&Ђ#

0џџџџџџџџџ
 Ю
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15899g`abc7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "&Ђ#

0џџџџџџџџџ
 Ю
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15917g`abc7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "&Ђ#

0џџџџџџџџџ
 Љ
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15061]`abc:Ђ7
0Ђ-
# 
	latent_inџџџџџџџџџ
p

 
Њ "џџџџџџџџџЉ
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15089]`abc:Ђ7
0Ђ-
# 
	latent_inџџџџџџџџџ
p 

 
Њ "џџџџџџџџџІ
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15930Z`abc7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџІ
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15943Z`abc7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџІ
E__inference_decoder_d1_layer_call_and_return_conditional_losses_16053]`a/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 ~
*__inference_decoder_d1_layer_call_fn_16062P`a/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЋ
I__inference_decoder_output_layer_call_and_return_conditional_losses_16083^bc0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
.__inference_decoder_output_layer_call_fn_16092Qbc0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_14834Е/0AB;<?Ђ<
5Ђ2
(%
encoder_inputџџџџџџџџџ
p

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_14857Е/0AB;<?Ђ<
5Ђ2
(%
encoder_inputџџџџџџџџџ
p 

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15803Ў/0AB;<8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 
c__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15839Ў/0AB;<8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p 

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 ђ
H__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_14902Ѕ/0AB;<?Ђ<
5Ђ2
(%
encoder_inputџџџџџџџџџ
p

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџђ
H__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_14946Ѕ/0AB;<?Ђ<
5Ђ2
(%
encoder_inputџџџџџџџџџ
p 

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџы
H__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15860/0AB;<8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџы
H__inference_enc-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15881/0AB;<8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p 

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџЇ
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_15953^/00Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_ref_enc_d1_layer_call_fn_15962Q/00Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџШ
@__inference_ref_z_layer_call_and_return_conditional_losses_16037ZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
%__inference_ref_z_layer_call_fn_16043vZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "џџџџџџџџџЉ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_15982];<0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
-__inference_ref_z_log_var_layer_call_fn_15991P;<0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_16001]AB0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
*__inference_ref_z_mean_layer_call_fn_16010PAB0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџє
#__inference_signature_wrapper_15535Ь
/0AB;<`abcHЂE
Ђ 
>Њ;
9
encoder_input(%
encoder_inputџџџџџџџџџ"tЊq
o
(dec-ref-512-16-msle-g-relu-fashion_mnistC@
(dec-ref-512-16-msle-g-relu-fashion_mnistџџџџџџџџџы
c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15240
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
1/0 ы
c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15309
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
1/0 у
c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15625|
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
1/0 у
c__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_15715|
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
1/0 Д
H__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15405h
/0AB;<`abc?Ђ<
5Ђ2
(%
encoder_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџД
H__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15500h
/0AB;<`abc?Ђ<
5Ђ2
(%
encoder_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ­
H__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15741a
/0AB;<`abc8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ­
H__inference_vae-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_15767a
/0AB;<`abc8Ђ5
.Ђ+
!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ