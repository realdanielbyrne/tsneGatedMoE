╠і
ПБ
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
dtypetypeѕ
Й
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
executor_typestring ѕ
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.4.0-dev202007052v1.12.1-35711-g8202ae9d5c8тк
ђ
ref_enc_d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
љђ*"
shared_nameref_enc_d1/kernel
y
%ref_enc_d1/kernel/Read/ReadVariableOpReadVariableOpref_enc_d1/kernel* 
_output_shapes
:
љђ*
dtype0
w
ref_enc_d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ* 
shared_nameref_enc_d1/bias
p
#ref_enc_d1/bias/Read/ReadVariableOpReadVariableOpref_enc_d1/bias*
_output_shapes	
:ђ*
dtype0

ref_z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*"
shared_nameref_z_mean/kernel
x
%ref_z_mean/kernel/Read/ReadVariableOpReadVariableOpref_z_mean/kernel*
_output_shapes
:	ђ*
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
Ё
ref_z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*%
shared_nameref_z_log_var/kernel
~
(ref_z_log_var/kernel/Read/ReadVariableOpReadVariableOpref_z_log_var/kernel*
_output_shapes
:	ђ*
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

NoOpNoOp
Ы
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Г
valueБBа BЎ
І
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
R
#regularization_losses
$trainable_variables
%	variables
&	keras_api
R
'regularization_losses
(trainable_variables
)	variables
*	keras_api
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
Г

+layers
,layer_regularization_losses
regularization_losses
	trainable_variables
-non_trainable_variables
.metrics

	variables
/layer_metrics
 
][
VARIABLE_VALUEref_enc_d1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEref_enc_d1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Г
0layer_regularization_losses

1layers
regularization_losses
trainable_variables
2non_trainable_variables
3metrics
	variables
4layer_metrics
 
 
 
Г
5layer_regularization_losses

6layers
regularization_losses
trainable_variables
7non_trainable_variables
8metrics
	variables
9layer_metrics
][
VARIABLE_VALUEref_z_mean/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEref_z_mean/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Г
:layer_regularization_losses

;layers
regularization_losses
trainable_variables
<non_trainable_variables
=metrics
	variables
>layer_metrics
`^
VARIABLE_VALUEref_z_log_var/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEref_z_log_var/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Г
?layer_regularization_losses

@layers
regularization_losses
 trainable_variables
Anon_trainable_variables
Bmetrics
!	variables
Clayer_metrics
 
 
 
Г
Dlayer_regularization_losses

Elayers
#regularization_losses
$trainable_variables
Fnon_trainable_variables
Gmetrics
%	variables
Hlayer_metrics
 
 
 
Г
Ilayer_regularization_losses

Jlayers
'regularization_losses
(trainable_variables
Knon_trainable_variables
Lmetrics
)	variables
Mlayer_metrics
1
0
1
2
3
4
5
6
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
 
 
 
 
 
 
ѓ
serving_default_encoder_inputPlaceholder*(
_output_shapes
:         љ*
dtype0*
shape:         љ
Я
StatefulPartitionedCallStatefulPartitionedCallserving_default_encoder_inputref_enc_d1/kernelref_enc_d1/biasref_z_mean/kernelref_z_mean/biasref_z_log_var/kernelref_z_log_var/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *,
f'R%
#__inference_signature_wrapper_16907
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ї
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%ref_enc_d1/kernel/Read/ReadVariableOp#ref_enc_d1/bias/Read/ReadVariableOp%ref_z_mean/kernel/Read/ReadVariableOp#ref_z_mean/bias/Read/ReadVariableOp(ref_z_log_var/kernel/Read/ReadVariableOp&ref_z_log_var/bias/Read/ReadVariableOpConst*
Tin

2*
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
GPU2*0J 8ѓ *'
f"R 
__inference__traced_save_17181
љ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameref_enc_d1/kernelref_enc_d1/biasref_z_mean/kernelref_z_mean/biasref_z_log_var/kernelref_z_log_var/bias*
Tin
	2*
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
GPU2*0J 8ѓ **
f%R#
!__inference__traced_restore_17209▒ќ
ъ
n
%__inference_ref_z_layer_call_fn_17115
inputs_0
inputs_1
identityѕбStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_167312
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         :         22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
т

*__inference_ref_enc_d1_layer_call_fn_17045

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_166342
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         љ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
н
░
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_16699

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:::P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
║
\
@__inference_floor_layer_call_and_return_conditional_losses_17124

inputs
identityK
AbsAbsinputs*
T0*'
_output_shapes
:         2
AbsU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
Less/y`
LessLessAbs:y:0Less/y:output:0*
T0*'
_output_shapes
:         2
Less_

zeros_like	ZerosLikeinputs*
T0*'
_output_shapes
:         2

zeros_liket
SelectV2SelectV2Less:z:0zeros_like:y:0inputs*
T0*'
_output_shapes
:         2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ж
ѓ
-__inference_ref_z_log_var_layer_call_fn_17093

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_166992
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ю=
█
 __inference__wrapped_model_16620
encoder_inputP
Lenc_ref_512_16_msle_g_relu_mnist_f_ref_enc_d1_matmul_readvariableop_resourceQ
Menc_ref_512_16_msle_g_relu_mnist_f_ref_enc_d1_biasadd_readvariableop_resourceP
Lenc_ref_512_16_msle_g_relu_mnist_f_ref_z_mean_matmul_readvariableop_resourceQ
Menc_ref_512_16_msle_g_relu_mnist_f_ref_z_mean_biasadd_readvariableop_resourceS
Oenc_ref_512_16_msle_g_relu_mnist_f_ref_z_log_var_matmul_readvariableop_resourceT
Penc_ref_512_16_msle_g_relu_mnist_f_ref_z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2ѕЎ
Cenc-ref-512-16-msle-g-relu-mnist-f/ref_enc_d1/MatMul/ReadVariableOpReadVariableOpLenc_ref_512_16_msle_g_relu_mnist_f_ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
љђ*
dtype02E
Cenc-ref-512-16-msle-g-relu-mnist-f/ref_enc_d1/MatMul/ReadVariableOpЁ
4enc-ref-512-16-msle-g-relu-mnist-f/ref_enc_d1/MatMulMatMulencoder_inputKenc-ref-512-16-msle-g-relu-mnist-f/ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ26
4enc-ref-512-16-msle-g-relu-mnist-f/ref_enc_d1/MatMulЌ
Denc-ref-512-16-msle-g-relu-mnist-f/ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOpMenc_ref_512_16_msle_g_relu_mnist_f_ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02F
Denc-ref-512-16-msle-g-relu-mnist-f/ref_enc_d1/BiasAdd/ReadVariableOp║
5enc-ref-512-16-msle-g-relu-mnist-f/ref_enc_d1/BiasAddBiasAdd>enc-ref-512-16-msle-g-relu-mnist-f/ref_enc_d1/MatMul:product:0Lenc-ref-512-16-msle-g-relu-mnist-f/ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ27
5enc-ref-512-16-msle-g-relu-mnist-f/ref_enc_d1/BiasAddс
2enc-ref-512-16-msle-g-relu-mnist-f/activation/ReluRelu>enc-ref-512-16-msle-g-relu-mnist-f/ref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ24
2enc-ref-512-16-msle-g-relu-mnist-f/activation/Reluў
Cenc-ref-512-16-msle-g-relu-mnist-f/ref_z_mean/MatMul/ReadVariableOpReadVariableOpLenc_ref_512_16_msle_g_relu_mnist_f_ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02E
Cenc-ref-512-16-msle-g-relu-mnist-f/ref_z_mean/MatMul/ReadVariableOpи
4enc-ref-512-16-msle-g-relu-mnist-f/ref_z_mean/MatMulMatMul@enc-ref-512-16-msle-g-relu-mnist-f/activation/Relu:activations:0Kenc-ref-512-16-msle-g-relu-mnist-f/ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         26
4enc-ref-512-16-msle-g-relu-mnist-f/ref_z_mean/MatMulќ
Denc-ref-512-16-msle-g-relu-mnist-f/ref_z_mean/BiasAdd/ReadVariableOpReadVariableOpMenc_ref_512_16_msle_g_relu_mnist_f_ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02F
Denc-ref-512-16-msle-g-relu-mnist-f/ref_z_mean/BiasAdd/ReadVariableOp╣
5enc-ref-512-16-msle-g-relu-mnist-f/ref_z_mean/BiasAddBiasAdd>enc-ref-512-16-msle-g-relu-mnist-f/ref_z_mean/MatMul:product:0Lenc-ref-512-16-msle-g-relu-mnist-f/ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         27
5enc-ref-512-16-msle-g-relu-mnist-f/ref_z_mean/BiasAddА
Fenc-ref-512-16-msle-g-relu-mnist-f/ref_z_log_var/MatMul/ReadVariableOpReadVariableOpOenc_ref_512_16_msle_g_relu_mnist_f_ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02H
Fenc-ref-512-16-msle-g-relu-mnist-f/ref_z_log_var/MatMul/ReadVariableOp└
7enc-ref-512-16-msle-g-relu-mnist-f/ref_z_log_var/MatMulMatMul@enc-ref-512-16-msle-g-relu-mnist-f/activation/Relu:activations:0Nenc-ref-512-16-msle-g-relu-mnist-f/ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         29
7enc-ref-512-16-msle-g-relu-mnist-f/ref_z_log_var/MatMulЪ
Genc-ref-512-16-msle-g-relu-mnist-f/ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOpPenc_ref_512_16_msle_g_relu_mnist_f_ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02I
Genc-ref-512-16-msle-g-relu-mnist-f/ref_z_log_var/BiasAdd/ReadVariableOp┼
8enc-ref-512-16-msle-g-relu-mnist-f/ref_z_log_var/BiasAddBiasAddAenc-ref-512-16-msle-g-relu-mnist-f/ref_z_log_var/MatMul:product:0Oenc-ref-512-16-msle-g-relu-mnist-f/ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2:
8enc-ref-512-16-msle-g-relu-mnist-f/ref_z_log_var/BiasAdd╬
.enc-ref-512-16-msle-g-relu-mnist-f/ref_z/ShapeShape>enc-ref-512-16-msle-g-relu-mnist-f/ref_z_mean/BiasAdd:output:0*
T0*
_output_shapes
:20
.enc-ref-512-16-msle-g-relu-mnist-f/ref_z/Shape┐
;enc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2=
;enc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normal/mean├
=enc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2?
=enc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normal/stddevф
Kenc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normal/RandomStandardNormalRandomStandardNormal7enc-ref-512-16-msle-g-relu-mnist-f/ref_z/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02M
Kenc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normal/RandomStandardNormal¤
:enc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normal/mulMulTenc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normal/RandomStandardNormal:output:0Fenc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:         2<
:enc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normal/mul»
6enc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normalAdd>enc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normal/mul:z:0Denc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:         28
6enc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normalЦ
.enc-ref-512-16-msle-g-relu-mnist-f/ref_z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?20
.enc-ref-512-16-msle-g-relu-mnist-f/ref_z/mul/xЉ
,enc-ref-512-16-msle-g-relu-mnist-f/ref_z/mulMul7enc-ref-512-16-msle-g-relu-mnist-f/ref_z/mul/x:output:0Aenc-ref-512-16-msle-g-relu-mnist-f/ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         2.
,enc-ref-512-16-msle-g-relu-mnist-f/ref_z/mulК
,enc-ref-512-16-msle-g-relu-mnist-f/ref_z/ExpExp0enc-ref-512-16-msle-g-relu-mnist-f/ref_z/mul:z:0*
T0*'
_output_shapes
:         2.
,enc-ref-512-16-msle-g-relu-mnist-f/ref_z/ExpЄ
.enc-ref-512-16-msle-g-relu-mnist-f/ref_z/mul_1Mul0enc-ref-512-16-msle-g-relu-mnist-f/ref_z/Exp:y:0:enc-ref-512-16-msle-g-relu-mnist-f/ref_z/random_normal:z:0*
T0*'
_output_shapes
:         20
.enc-ref-512-16-msle-g-relu-mnist-f/ref_z/mul_1І
,enc-ref-512-16-msle-g-relu-mnist-f/ref_z/addAddV2>enc-ref-512-16-msle-g-relu-mnist-f/ref_z_mean/BiasAdd:output:02enc-ref-512-16-msle-g-relu-mnist-f/ref_z/mul_1:z:0*
T0*'
_output_shapes
:         2.
,enc-ref-512-16-msle-g-relu-mnist-f/ref_z/addё
IdentityIdentity0enc-ref-512-16-msle-g-relu-mnist-f/ref_z/add:z:0*
T0*'
_output_shapes
:         2

IdentityЎ

Identity_1IdentityAenc-ref-512-16-msle-g-relu-mnist-f/ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity_1ќ

Identity_2Identity>enc-ref-512-16-msle-g-relu-mnist-f/ref_z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:         љ:::::::W S
(
_output_shapes
:         љ
'
_user_specified_nameencoder_input
Ї
A
%__inference_floor_layer_call_fn_17133

inputs
identity┴
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_floor_layer_call_and_return_conditional_losses_167492
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
е
Ќ
__inference__traced_save_17181
file_prefix0
,savev2_ref_enc_d1_kernel_read_readvariableop.
*savev2_ref_enc_d1_bias_read_readvariableop0
,savev2_ref_z_mean_kernel_read_readvariableop.
*savev2_ref_z_mean_bias_read_readvariableop3
/savev2_ref_z_log_var_kernel_read_readvariableop1
-savev2_ref_z_log_var_bias_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_51e969b2b208485c837a4d20b72d27f6/part2	
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameв
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*§
valueзB­B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesќ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slicesн
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_ref_enc_d1_kernel_read_readvariableop*savev2_ref_enc_d1_bias_read_readvariableop,savev2_ref_z_mean_kernel_read_readvariableop*savev2_ref_z_mean_bias_read_readvariableop/savev2_ref_z_log_var_kernel_read_readvariableop-savev2_ref_z_log_var_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*L
_input_shapes;
9: :
љђ:ђ:	ђ::	ђ:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
љђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ: 

_output_shapes
::%!

_output_shapes
:	ђ: 

_output_shapes
::

_output_shapes
: 
╠
з
B__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_fn_17026

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2ѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *f
faR_
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_168652
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identityњ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1њ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:         љ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
Ё
█
#__inference_signature_wrapper_16907
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2ѕбStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *)
f$R"
 __inference__wrapped_model_166202
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identityњ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1њ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:         љ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:         љ
'
_user_specified_nameencoder_input
Џ
F
*__inference_activation_layer_call_fn_17055

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_166552
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ы!
И
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_16865

inputs
ref_enc_d1_16844
ref_enc_d1_16846
ref_z_mean_16850
ref_z_mean_16852
ref_z_log_var_16855
ref_z_log_var_16857
identity

identity_1

identity_2ѕб"ref_enc_d1/StatefulPartitionedCallбref_z/StatefulPartitionedCallб%ref_z_log_var/StatefulPartitionedCallб"ref_z_mean/StatefulPartitionedCallЪ
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallinputsref_enc_d1_16844ref_enc_d1_16846*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_166342$
"ref_enc_d1/StatefulPartitionedCallѓ
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_166552
activation/PartitionedCall╗
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_16850ref_z_mean_16852*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_166732$
"ref_z_mean/StatefulPartitionedCall╩
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_16855ref_z_log_var_16857*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_166992'
%ref_z_log_var/StatefulPartitionedCall╗
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_167312
ref_z/StatefulPartitionedCallь
floor/PartitionedCallPartitionedCall&ref_z/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_floor_layer_call_and_return_conditional_losses_167532
floor/PartitionedCallЉ
IdentityIdentity+ref_z_mean/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identityў

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1ѕ

Identity_2Identityfloor/PartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:         љ::::::2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2>
ref_z/StatefulPartitionedCallref_z/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
н
░
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_17084

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:::P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
с

*__inference_ref_z_mean_layer_call_fn_17074

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_166732
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ї
A
%__inference_floor_layer_call_fn_17138

inputs
identity┴
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_floor_layer_call_and_return_conditional_losses_167532
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
м
\
@__inference_floor_layer_call_and_return_conditional_losses_16753

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
и
a
E__inference_activation_layer_call_and_return_conditional_losses_16655

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:         ђ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
р
Щ
B__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_fn_16839
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2ѕбStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *f
faR_
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_168202
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identityњ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1њ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:         љ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:         љ
'
_user_specified_nameencoder_input
И
й
!__inference__traced_restore_17209
file_prefix&
"assignvariableop_ref_enc_d1_kernel&
"assignvariableop_1_ref_enc_d1_bias(
$assignvariableop_2_ref_z_mean_kernel&
"assignvariableop_3_ref_z_mean_bias+
'assignvariableop_4_ref_z_log_var_kernel)
%assignvariableop_5_ref_z_log_var_bias

identity_7ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5ы
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*§
valueзB­B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesю
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices╬
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityА
AssignVariableOpAssignVariableOp"assignvariableop_ref_enc_d1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Д
AssignVariableOp_1AssignVariableOp"assignvariableop_1_ref_enc_d1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Е
AssignVariableOp_2AssignVariableOp$assignvariableop_2_ref_z_mean_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Д
AssignVariableOp_3AssignVariableOp"assignvariableop_3_ref_z_mean_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4г
AssignVariableOp_4AssignVariableOp'assignvariableop_4_ref_z_log_var_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ф
AssignVariableOp_5AssignVariableOp%assignvariableop_5_ref_z_log_var_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpС

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6о

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Л
Г
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_16673

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:::P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╠
з
B__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_fn_17005

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2ѕбStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *f
faR_
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_168202
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identityњ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1њ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:         љ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
м
\
@__inference_floor_layer_call_and_return_conditional_losses_17128

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Е
o
@__inference_ref_z_layer_call_and_return_conditional_losses_17109
inputs_0
inputs_1
identityѕF
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
 *  ђ?2
random_normal/stddev»
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:         *
dtype02$
"random_normal/RandomStandardNormalФ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:         2
random_normal/mulІ
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:         2
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
:         2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:         2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:         2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:         2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
о
Г
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_16634

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
љђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         љ:::P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
є"
┐
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_16769
encoder_input
ref_enc_d1_16645
ref_enc_d1_16647
ref_z_mean_16684
ref_z_mean_16686
ref_z_log_var_16710
ref_z_log_var_16712
identity

identity_1

identity_2ѕб"ref_enc_d1/StatefulPartitionedCallбref_z/StatefulPartitionedCallб%ref_z_log_var/StatefulPartitionedCallб"ref_z_mean/StatefulPartitionedCallд
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputref_enc_d1_16645ref_enc_d1_16647*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_166342$
"ref_enc_d1/StatefulPartitionedCallѓ
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_166552
activation/PartitionedCall╗
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_16684ref_z_mean_16686*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_166732$
"ref_z_mean/StatefulPartitionedCall╩
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_16710ref_z_log_var_16712*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_166992'
%ref_z_log_var/StatefulPartitionedCall╗
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_167312
ref_z/StatefulPartitionedCallь
floor/PartitionedCallPartitionedCall&ref_z/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_floor_layer_call_and_return_conditional_losses_167492
floor/PartitionedCallЉ
IdentityIdentity+ref_z_mean/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identityў

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1ѕ

Identity_2Identityfloor/PartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:         љ::::::2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2>
ref_z/StatefulPartitionedCallref_z/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:W S
(
_output_shapes
:         љ
'
_user_specified_nameencoder_input
Ъ
m
@__inference_ref_z_layer_call_and_return_conditional_losses_16731

inputs
inputs_1
identityѕD
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
 *  ђ?2
random_normal/stddev»
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:         *
dtype02$
"random_normal/RandomStandardNormalФ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:         2
random_normal/mulІ
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:         2
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
:         2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:         2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:         2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:         2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:         :         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
▓%
┐
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_16984

inputs-
)ref_enc_d1_matmul_readvariableop_resource.
*ref_enc_d1_biasadd_readvariableop_resource-
)ref_z_mean_matmul_readvariableop_resource.
*ref_z_mean_biasadd_readvariableop_resource0
,ref_z_log_var_matmul_readvariableop_resource1
-ref_z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2ѕ░
 ref_enc_d1/MatMul/ReadVariableOpReadVariableOp)ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
љђ*
dtype02"
 ref_enc_d1/MatMul/ReadVariableOpЋ
ref_enc_d1/MatMulMatMulinputs(ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
ref_enc_d1/MatMul«
!ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOp*ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02#
!ref_enc_d1/BiasAdd/ReadVariableOp«
ref_enc_d1/BiasAddBiasAddref_enc_d1/MatMul:product:0)ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
ref_enc_d1/BiasAddz
activation/ReluReluref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
activation/Relu»
 ref_z_mean/MatMul/ReadVariableOpReadVariableOp)ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02"
 ref_z_mean/MatMul/ReadVariableOpФ
ref_z_mean/MatMulMatMulactivation/Relu:activations:0(ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
ref_z_mean/MatMulГ
!ref_z_mean/BiasAdd/ReadVariableOpReadVariableOp*ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ref_z_mean/BiasAdd/ReadVariableOpГ
ref_z_mean/BiasAddBiasAddref_z_mean/MatMul:product:0)ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
ref_z_mean/BiasAddИ
#ref_z_log_var/MatMul/ReadVariableOpReadVariableOp,ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02%
#ref_z_log_var/MatMul/ReadVariableOp┤
ref_z_log_var/MatMulMatMulactivation/Relu:activations:0+ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
ref_z_log_var/MatMulХ
$ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOp-ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ref_z_log_var/BiasAdd/ReadVariableOp╣
ref_z_log_var/BiasAddBiasAddref_z_log_var/MatMul:product:0,ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
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
 *  ђ?2
ref_z/random_normal/stddev┴
(ref_z/random_normal/RandomStandardNormalRandomStandardNormalref_z/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02*
(ref_z/random_normal/RandomStandardNormal├
ref_z/random_normal/mulMul1ref_z/random_normal/RandomStandardNormal:output:0#ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:         2
ref_z/random_normal/mulБ
ref_z/random_normalAddref_z/random_normal/mul:z:0!ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:         2
ref_z/random_normal_
ref_z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
ref_z/mul/xЁ
	ref_z/mulMulref_z/mul/x:output:0ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         2
	ref_z/mul^
	ref_z/ExpExpref_z/mul:z:0*
T0*'
_output_shapes
:         2
	ref_z/Exp{
ref_z/mul_1Mulref_z/Exp:y:0ref_z/random_normal:z:0*
T0*'
_output_shapes
:         2
ref_z/mul_1
	ref_z/addAddV2ref_z_mean/BiasAdd:output:0ref_z/mul_1:z:0*
T0*'
_output_shapes
:         2
	ref_z/addo
IdentityIdentityref_z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identityv

Identity_1Identityref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity_1e

Identity_2Identityref_z/add:z:0*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:         љ:::::::P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
║
\
@__inference_floor_layer_call_and_return_conditional_losses_16749

inputs
identityK
AbsAbsinputs*
T0*'
_output_shapes
:         2
AbsU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
Less/y`
LessLessAbs:y:0Less/y:output:0*
T0*'
_output_shapes
:         2
Less_

zeros_like	ZerosLikeinputs*
T0*'
_output_shapes
:         2

zeros_liket
SelectV2SelectV2Less:z:0zeros_like:y:0inputs*
T0*'
_output_shapes
:         2

SelectV2e
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
о
Г
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_17036

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
љђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         љ:::P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
ы!
И
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_16820

inputs
ref_enc_d1_16799
ref_enc_d1_16801
ref_z_mean_16805
ref_z_mean_16807
ref_z_log_var_16810
ref_z_log_var_16812
identity

identity_1

identity_2ѕб"ref_enc_d1/StatefulPartitionedCallбref_z/StatefulPartitionedCallб%ref_z_log_var/StatefulPartitionedCallб"ref_z_mean/StatefulPartitionedCallЪ
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallinputsref_enc_d1_16799ref_enc_d1_16801*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_166342$
"ref_enc_d1/StatefulPartitionedCallѓ
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_166552
activation/PartitionedCall╗
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_16805ref_z_mean_16807*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_166732$
"ref_z_mean/StatefulPartitionedCall╩
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_16810ref_z_log_var_16812*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_166992'
%ref_z_log_var/StatefulPartitionedCall╗
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_167312
ref_z/StatefulPartitionedCallь
floor/PartitionedCallPartitionedCall&ref_z/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_floor_layer_call_and_return_conditional_losses_167492
floor/PartitionedCallЉ
IdentityIdentity+ref_z_mean/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identityў

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1ѕ

Identity_2Identityfloor/PartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:         љ::::::2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2>
ref_z/StatefulPartitionedCallref_z/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
є"
┐
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_16793
encoder_input
ref_enc_d1_16772
ref_enc_d1_16774
ref_z_mean_16778
ref_z_mean_16780
ref_z_log_var_16783
ref_z_log_var_16785
identity

identity_1

identity_2ѕб"ref_enc_d1/StatefulPartitionedCallбref_z/StatefulPartitionedCallб%ref_z_log_var/StatefulPartitionedCallб"ref_z_mean/StatefulPartitionedCallд
"ref_enc_d1/StatefulPartitionedCallStatefulPartitionedCallencoder_inputref_enc_d1_16772ref_enc_d1_16774*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_166342$
"ref_enc_d1/StatefulPartitionedCallѓ
activation/PartitionedCallPartitionedCall+ref_enc_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_166552
activation/PartitionedCall╗
"ref_z_mean/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_mean_16778ref_z_mean_16780*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_166732$
"ref_z_mean/StatefulPartitionedCall╩
%ref_z_log_var/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0ref_z_log_var_16783ref_z_log_var_16785*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_166992'
%ref_z_log_var/StatefulPartitionedCall╗
ref_z/StatefulPartitionedCallStatefulPartitionedCall+ref_z_mean/StatefulPartitionedCall:output:0.ref_z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_ref_z_layer_call_and_return_conditional_losses_167312
ref_z/StatefulPartitionedCallь
floor/PartitionedCallPartitionedCall&ref_z/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_floor_layer_call_and_return_conditional_losses_167532
floor/PartitionedCallЉ
IdentityIdentity+ref_z_mean/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identityў

Identity_1Identity.ref_z_log_var/StatefulPartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1ѕ

Identity_2Identityfloor/PartitionedCall:output:0#^ref_enc_d1/StatefulPartitionedCall^ref_z/StatefulPartitionedCall&^ref_z_log_var/StatefulPartitionedCall#^ref_z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:         љ::::::2H
"ref_enc_d1/StatefulPartitionedCall"ref_enc_d1/StatefulPartitionedCall2>
ref_z/StatefulPartitionedCallref_z/StatefulPartitionedCall2N
%ref_z_log_var/StatefulPartitionedCall%ref_z_log_var/StatefulPartitionedCall2H
"ref_z_mean/StatefulPartitionedCall"ref_z_mean/StatefulPartitionedCall:W S
(
_output_shapes
:         љ
'
_user_specified_nameencoder_input
Ѓ*
┐
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_16948

inputs-
)ref_enc_d1_matmul_readvariableop_resource.
*ref_enc_d1_biasadd_readvariableop_resource-
)ref_z_mean_matmul_readvariableop_resource.
*ref_z_mean_biasadd_readvariableop_resource0
,ref_z_log_var_matmul_readvariableop_resource1
-ref_z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2ѕ░
 ref_enc_d1/MatMul/ReadVariableOpReadVariableOp)ref_enc_d1_matmul_readvariableop_resource* 
_output_shapes
:
љђ*
dtype02"
 ref_enc_d1/MatMul/ReadVariableOpЋ
ref_enc_d1/MatMulMatMulinputs(ref_enc_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
ref_enc_d1/MatMul«
!ref_enc_d1/BiasAdd/ReadVariableOpReadVariableOp*ref_enc_d1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02#
!ref_enc_d1/BiasAdd/ReadVariableOp«
ref_enc_d1/BiasAddBiasAddref_enc_d1/MatMul:product:0)ref_enc_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
ref_enc_d1/BiasAddz
activation/ReluReluref_enc_d1/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
activation/Relu»
 ref_z_mean/MatMul/ReadVariableOpReadVariableOp)ref_z_mean_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02"
 ref_z_mean/MatMul/ReadVariableOpФ
ref_z_mean/MatMulMatMulactivation/Relu:activations:0(ref_z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
ref_z_mean/MatMulГ
!ref_z_mean/BiasAdd/ReadVariableOpReadVariableOp*ref_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!ref_z_mean/BiasAdd/ReadVariableOpГ
ref_z_mean/BiasAddBiasAddref_z_mean/MatMul:product:0)ref_z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
ref_z_mean/BiasAddИ
#ref_z_log_var/MatMul/ReadVariableOpReadVariableOp,ref_z_log_var_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02%
#ref_z_log_var/MatMul/ReadVariableOp┤
ref_z_log_var/MatMulMatMulactivation/Relu:activations:0+ref_z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
ref_z_log_var/MatMulХ
$ref_z_log_var/BiasAdd/ReadVariableOpReadVariableOp-ref_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$ref_z_log_var/BiasAdd/ReadVariableOp╣
ref_z_log_var/BiasAddBiasAddref_z_log_var/MatMul:product:0,ref_z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
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
 *  ђ?2
ref_z/random_normal/stddev┴
(ref_z/random_normal/RandomStandardNormalRandomStandardNormalref_z/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02*
(ref_z/random_normal/RandomStandardNormal├
ref_z/random_normal/mulMul1ref_z/random_normal/RandomStandardNormal:output:0#ref_z/random_normal/stddev:output:0*
T0*'
_output_shapes
:         2
ref_z/random_normal/mulБ
ref_z/random_normalAddref_z/random_normal/mul:z:0!ref_z/random_normal/mean:output:0*
T0*'
_output_shapes
:         2
ref_z/random_normal_
ref_z/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
ref_z/mul/xЁ
	ref_z/mulMulref_z/mul/x:output:0ref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         2
	ref_z/mul^
	ref_z/ExpExpref_z/mul:z:0*
T0*'
_output_shapes
:         2
	ref_z/Exp{
ref_z/mul_1Mulref_z/Exp:y:0ref_z/random_normal:z:0*
T0*'
_output_shapes
:         2
ref_z/mul_1
	ref_z/addAddV2ref_z_mean/BiasAdd:output:0ref_z/mul_1:z:0*
T0*'
_output_shapes
:         2
	ref_z/add^
	floor/AbsAbsref_z/add:z:0*
T0*'
_output_shapes
:         2
	floor/Absa
floor/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
floor/Less/yx

floor/LessLessfloor/Abs:y:0floor/Less/y:output:0*
T0*'
_output_shapes
:         2

floor/Lessr
floor/zeros_like	ZerosLikeref_z/add:z:0*
T0*'
_output_shapes
:         2
floor/zeros_likeЊ
floor/SelectV2SelectV2floor/Less:z:0floor/zeros_like:y:0ref_z/add:z:0*
T0*'
_output_shapes
:         2
floor/SelectV2o
IdentityIdentityref_z_mean/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identityv

Identity_1Identityref_z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity_1o

Identity_2Identityfloor/SelectV2:output:0*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:         љ:::::::P L
(
_output_shapes
:         љ
 
_user_specified_nameinputs
и
a
E__inference_activation_layer_call_and_return_conditional_losses_17050

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:         ђ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
р
Щ
B__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_fn_16884
encoder_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1

identity_2ѕбStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *f
faR_
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_168652
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identityњ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1њ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*?
_input_shapes.
,:         љ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:         љ
'
_user_specified_nameencoder_input
Л
Г
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_17065

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:::P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs"─L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*И
serving_defaultц
H
encoder_input7
serving_default_encoder_input:0         љ9
floor0
StatefulPartitionedCall:0         A
ref_z_log_var0
StatefulPartitionedCall:1         >

ref_z_mean0
StatefulPartitionedCall:2         tensorflow/serving/predict:Ћ╗
╔,
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
N__call__
O_default_save_signature
*P&call_and_return_all_conditional_losses"С)
_tf_keras_network╚){"class_name": "Functional", "name": "enc-ref-512-16-msle-g-relu-mnist-f", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "enc-ref-512-16-msle-g-relu-mnist-f", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}, "name": "ref_z", "inbound_nodes": [[["ref_z_mean", 0, 0, {}], ["ref_z_log_var", 0, 0, {}]]]}, {"class_name": "Floor", "config": {"zero_point": 0.1, "mean_on_eval": true}, "name": "floor", "inbound_nodes": [[["ref_z", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["ref_z_mean", 0, 0], ["ref_z_log_var", 0, 0], ["floor", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "enc-ref-512-16-msle-g-relu-mnist-f", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}, "name": "encoder_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_enc_d1", "inbound_nodes": [[["encoder_input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["ref_enc_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_mean", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "ref_z_log_var", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}, "name": "ref_z", "inbound_nodes": [[["ref_z_mean", 0, 0, {}], ["ref_z_log_var", 0, 0, {}]]]}, {"class_name": "Floor", "config": {"zero_point": 0.1, "mean_on_eval": true}, "name": "floor", "inbound_nodes": [[["ref_z", 0, 0, {}]]]}], "input_layers": [["encoder_input", 0, 0]], "output_layers": [["ref_z_mean", 0, 0], ["ref_z_log_var", 0, 0], ["floor", 0, 0]]}}}
щ"Ш
_tf_keras_input_layerо{"class_name": "InputLayer", "name": "encoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}}
ч

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"о
_tf_keras_layer╝{"class_name": "Dense", "name": "ref_enc_d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_enc_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
Л
regularization_losses
trainable_variables
	variables
	keras_api
S__call__
*T&call_and_return_all_conditional_losses"┬
_tf_keras_layerе{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
Щ

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
U__call__
*V&call_and_return_all_conditional_losses"Н
_tf_keras_layer╗{"class_name": "Dense", "name": "ref_z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
ђ

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
W__call__
*X&call_and_return_all_conditional_losses"█
_tf_keras_layer┴{"class_name": "Dense", "name": "ref_z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
»
#regularization_losses
$trainable_variables
%	variables
&	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"а
_tf_keras_layerє{"class_name": "Sampling", "name": "ref_z", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ref_z", "trainable": true, "dtype": "float32"}}
ю
'regularization_losses
(trainable_variables
)	variables
*	keras_api
[__call__
*\&call_and_return_all_conditional_losses"Ї
_tf_keras_layerз{"class_name": "Floor", "name": "floor", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"zero_point": 0.1, "mean_on_eval": true}}
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
╩

+layers
,layer_regularization_losses
regularization_losses
	trainable_variables
-non_trainable_variables
.metrics

	variables
/layer_metrics
N__call__
O_default_save_signature
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
,
]serving_default"
signature_map
%:#
љђ2ref_enc_d1/kernel
:ђ2ref_enc_d1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Г
0layer_regularization_losses

1layers
regularization_losses
trainable_variables
2non_trainable_variables
3metrics
	variables
4layer_metrics
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
5layer_regularization_losses

6layers
regularization_losses
trainable_variables
7non_trainable_variables
8metrics
	variables
9layer_metrics
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
$:"	ђ2ref_z_mean/kernel
:2ref_z_mean/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Г
:layer_regularization_losses

;layers
regularization_losses
trainable_variables
<non_trainable_variables
=metrics
	variables
>layer_metrics
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
':%	ђ2ref_z_log_var/kernel
 :2ref_z_log_var/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Г
?layer_regularization_losses

@layers
regularization_losses
 trainable_variables
Anon_trainable_variables
Bmetrics
!	variables
Clayer_metrics
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Dlayer_regularization_losses

Elayers
#regularization_losses
$trainable_variables
Fnon_trainable_variables
Gmetrics
%	variables
Hlayer_metrics
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Ilayer_regularization_losses

Jlayers
'regularization_losses
(trainable_variables
Knon_trainable_variables
Lmetrics
)	variables
Mlayer_metrics
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
Q
0
1
2
3
4
5
6"
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
о2М
B__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_fn_17026
B__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_fn_17005
B__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_fn_16839
B__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_fn_16884└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
т2Р
 __inference__wrapped_model_16620й
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *-б*
(і%
encoder_input         љ
┬2┐
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_16984
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_16769
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_16948
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_16793└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
н2Л
*__inference_ref_enc_d1_layer_call_fn_17045б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_17036б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_activation_layer_call_fn_17055б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_activation_layer_call_and_return_conditional_losses_17050б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_ref_z_mean_layer_call_fn_17074б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_17065б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_ref_z_log_var_layer_call_fn_17093б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_17084б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
¤2╠
%__inference_ref_z_layer_call_fn_17115б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ж2у
@__inference_ref_z_layer_call_and_return_conditional_losses_17109б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѕ2Ё
%__inference_floor_layer_call_fn_17138
%__inference_floor_layer_call_fn_17133┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Й2╗
@__inference_floor_layer_call_and_return_conditional_losses_17128
@__inference_floor_layer_call_and_return_conditional_losses_17124┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
8B6
#__inference_signature_wrapper_16907encoder_inputЁ
 __inference__wrapped_model_16620Я7б4
-б*
(і%
encoder_input         љ
ф "юфў
(
floorі
floor         
8
ref_z_log_var'і$
ref_z_log_var         
2

ref_z_mean$і!

ref_z_mean         Б
E__inference_activation_layer_call_and_return_conditional_losses_17050Z0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ {
*__inference_activation_layer_call_fn_17055M0б-
&б#
!і
inputs         ђ
ф "і         ђЌ
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_16769х?б<
5б2
(і%
encoder_input         љ
p

 
ф "jбg
`џ]
і
0/0         
і
0/1         
і
0/2         
џ Ќ
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_16793х?б<
5б2
(і%
encoder_input         љ
p 

 
ф "jбg
`џ]
і
0/0         
і
0/1         
і
0/2         
џ љ
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_16948«8б5
.б+
!і
inputs         љ
p

 
ф "jбg
`џ]
і
0/0         
і
0/1         
і
0/2         
џ љ
]__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_and_return_conditional_losses_16984«8б5
.б+
!і
inputs         љ
p 

 
ф "jбg
`џ]
і
0/0         
і
0/1         
і
0/2         
џ В
B__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_fn_16839Ц?б<
5б2
(і%
encoder_input         љ
p

 
ф "ZџW
і
0         
і
1         
і
2         В
B__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_fn_16884Ц?б<
5б2
(і%
encoder_input         љ
p 

 
ф "ZџW
і
0         
і
1         
і
2         т
B__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_fn_17005ъ8б5
.б+
!і
inputs         љ
p

 
ф "ZџW
і
0         
і
1         
і
2         т
B__inference_enc-ref-512-16-msle-g-relu-mnist-f_layer_call_fn_17026ъ8б5
.б+
!і
inputs         љ
p 

 
ф "ZџW
і
0         
і
1         
і
2         а
@__inference_floor_layer_call_and_return_conditional_losses_17124\3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ а
@__inference_floor_layer_call_and_return_conditional_losses_17128\3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ x
%__inference_floor_layer_call_fn_17133O3б0
)б&
 і
inputs         
p
ф "і         x
%__inference_floor_layer_call_fn_17138O3б0
)б&
 і
inputs         
p 
ф "і         Д
E__inference_ref_enc_d1_layer_call_and_return_conditional_losses_17036^0б-
&б#
!і
inputs         љ
ф "&б#
і
0         ђ
џ 
*__inference_ref_enc_d1_layer_call_fn_17045Q0б-
&б#
!і
inputs         љ
ф "і         ђ╚
@__inference_ref_z_layer_call_and_return_conditional_losses_17109ЃZбW
PбM
KџH
"і
inputs/0         
"і
inputs/1         
ф "%б"
і
0         
џ Ъ
%__inference_ref_z_layer_call_fn_17115vZбW
PбM
KџH
"і
inputs/0         
"і
inputs/1         
ф "і         Е
H__inference_ref_z_log_var_layer_call_and_return_conditional_losses_17084]0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         
џ Ђ
-__inference_ref_z_log_var_layer_call_fn_17093P0б-
&б#
!і
inputs         ђ
ф "і         д
E__inference_ref_z_mean_layer_call_and_return_conditional_losses_17065]0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         
џ ~
*__inference_ref_z_mean_layer_call_fn_17074P0б-
&б#
!і
inputs         ђ
ф "і         Ў
#__inference_signature_wrapper_16907ыHбE
б 
>ф;
9
encoder_input(і%
encoder_input         љ"юфў
(
floorі
floor         
8
ref_z_log_var'і$
ref_z_log_var         
2

ref_z_mean$і!

ref_z_mean         