Č
ÝŁ
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
ž
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
 "serve*2.3.0-dev202006102v1.12.1-33718-gabd113a7148ŠÍ
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
*&
shared_namedecoder_output/kernel

)decoder_output/kernel/Read/ReadVariableOpReadVariableOpdecoder_output/kernel* 
_output_shapes
:
*
dtype0

decoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namedecoder_output/bias
x
'decoder_output/bias/Read/ReadVariableOpReadVariableOpdecoder_output/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
Ú
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B
Ę
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
 
h


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
 


0
1
2
3


0
1
2
3
­
metrics
non_trainable_variables
regularization_losses
layer_metrics
trainable_variables
	variables

layers
layer_regularization_losses
 
][
VARIABLE_VALUEdecoder_d1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdecoder_d1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 


0
1


0
1
­
metrics
 non_trainable_variables
regularization_losses
!layer_metrics
trainable_variables
	variables

"layers
#layer_regularization_losses
 
 
 
­
$metrics
%non_trainable_variables
regularization_losses
&layer_metrics
trainable_variables
	variables

'layers
(layer_regularization_losses
a_
VARIABLE_VALUEdecoder_output/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEdecoder_output/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
)metrics
*non_trainable_variables
regularization_losses
+layer_metrics
trainable_variables
	variables

,layers
-layer_regularization_losses
 
 
 

0
1
2
3
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
|
serving_default_latent_inPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

StatefulPartitionedCallStatefulPartitionedCallserving_default_latent_indecoder_d1/kerneldecoder_d1/biasdecoder_output/kerneldecoder_output/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference_signature_wrapper_9780
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ş
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%decoder_d1/kernel/Read/ReadVariableOp#decoder_d1/bias/Read/ReadVariableOp)decoder_output/kernel/Read/ReadVariableOp'decoder_output/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*&
f!R
__inference__traced_save_9930
ĺ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedecoder_d1/kerneldecoder_d1/biasdecoder_output/kerneldecoder_output/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__traced_restore_9954ÄŤ

Ź
D__inference_decoder_d1_layer_call_and_return_conditional_losses_9636

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
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


__inference__wrapped_model_9622
	latent_inP
Ldec_ref_512_16_msle_g_relu_cifar10_decoder_d1_matmul_readvariableop_resourceQ
Mdec_ref_512_16_msle_g_relu_cifar10_decoder_d1_biasadd_readvariableop_resourceT
Pdec_ref_512_16_msle_g_relu_cifar10_decoder_output_matmul_readvariableop_resourceU
Qdec_ref_512_16_msle_g_relu_cifar10_decoder_output_biasadd_readvariableop_resource
identity
Cdec-ref-512-16-msle-g-relu-cifar10/decoder_d1/MatMul/ReadVariableOpReadVariableOpLdec_ref_512_16_msle_g_relu_cifar10_decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02E
Cdec-ref-512-16-msle-g-relu-cifar10/decoder_d1/MatMul/ReadVariableOp
4dec-ref-512-16-msle-g-relu-cifar10/decoder_d1/MatMulMatMul	latent_inKdec-ref-512-16-msle-g-relu-cifar10/decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙26
4dec-ref-512-16-msle-g-relu-cifar10/decoder_d1/MatMul
Ddec-ref-512-16-msle-g-relu-cifar10/decoder_d1/BiasAdd/ReadVariableOpReadVariableOpMdec_ref_512_16_msle_g_relu_cifar10_decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02F
Ddec-ref-512-16-msle-g-relu-cifar10/decoder_d1/BiasAdd/ReadVariableOpş
5dec-ref-512-16-msle-g-relu-cifar10/decoder_d1/BiasAddBiasAdd>dec-ref-512-16-msle-g-relu-cifar10/decoder_d1/MatMul:product:0Ldec-ref-512-16-msle-g-relu-cifar10/decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙27
5dec-ref-512-16-msle-g-relu-cifar10/decoder_d1/BiasAddç
4dec-ref-512-16-msle-g-relu-cifar10/activation_1/ReluRelu>dec-ref-512-16-msle-g-relu-cifar10/decoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙26
4dec-ref-512-16-msle-g-relu-cifar10/activation_1/ReluĽ
Gdec-ref-512-16-msle-g-relu-cifar10/decoder_output/MatMul/ReadVariableOpReadVariableOpPdec_ref_512_16_msle_g_relu_cifar10_decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02I
Gdec-ref-512-16-msle-g-relu-cifar10/decoder_output/MatMul/ReadVariableOpĆ
8dec-ref-512-16-msle-g-relu-cifar10/decoder_output/MatMulMatMulBdec-ref-512-16-msle-g-relu-cifar10/activation_1/Relu:activations:0Odec-ref-512-16-msle-g-relu-cifar10/decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2:
8dec-ref-512-16-msle-g-relu-cifar10/decoder_output/MatMulŁ
Hdec-ref-512-16-msle-g-relu-cifar10/decoder_output/BiasAdd/ReadVariableOpReadVariableOpQdec_ref_512_16_msle_g_relu_cifar10_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02J
Hdec-ref-512-16-msle-g-relu-cifar10/decoder_output/BiasAdd/ReadVariableOpĘ
9dec-ref-512-16-msle-g-relu-cifar10/decoder_output/BiasAddBiasAddBdec-ref-512-16-msle-g-relu-cifar10/decoder_output/MatMul:product:0Pdec-ref-512-16-msle-g-relu-cifar10/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2;
9dec-ref-512-16-msle-g-relu-cifar10/decoder_output/BiasAddř
9dec-ref-512-16-msle-g-relu-cifar10/decoder_output/SigmoidSigmoidBdec-ref-512-16-msle-g-relu-cifar10/decoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2;
9dec-ref-512-16-msle-g-relu-cifar10/decoder_output/Sigmoid
IdentityIdentity=dec-ref-512-16-msle-g-relu-cifar10/decoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙:::::R N
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	latent_in:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ť
¨
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_9708
	latent_in
decoder_d1_9696
decoder_d1_9698
decoder_output_9702
decoder_output_9704
identity˘"decoder_d1/StatefulPartitionedCall˘&decoder_output/StatefulPartitionedCall
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCall	latent_indecoder_d1_9696decoder_d1_9698*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_decoder_d1_layer_call_and_return_conditional_losses_96362$
"decoder_d1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_96572
activation_1/PartitionedCallÉ
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_9702decoder_output_9704*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_decoder_output_layer_call_and_return_conditional_losses_96762(
&decoder_output/StatefulPartitionedCallŇ
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:R N
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	latent_in:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¸
b
F__inference_activation_1_layer_call_and_return_conditional_losses_9657

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˘
Ľ
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_9726

inputs
decoder_d1_9714
decoder_d1_9716
decoder_output_9720
decoder_output_9722
identity˘"decoder_d1/StatefulPartitionedCall˘&decoder_output/StatefulPartitionedCall
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_d1_9714decoder_d1_9716*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_decoder_d1_layer_call_and_return_conditional_losses_96362$
"decoder_d1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_96572
activation_1/PartitionedCallÉ
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_9720decoder_output_9722*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_decoder_output_layer_call_and_return_conditional_losses_96762(
&decoder_output/StatefulPartitionedCallŇ
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ä
´
A__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_fn_9829

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*e
f`R^
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_97262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ń

"__inference_signature_wrapper_9780
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCall	latent_inunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__wrapped_model_96222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	latent_in:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
˘
Ľ
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_9754

inputs
decoder_d1_9742
decoder_d1_9744
decoder_output_9748
decoder_output_9750
identity˘"decoder_d1/StatefulPartitionedCall˘&decoder_output/StatefulPartitionedCall
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_d1_9742decoder_d1_9744*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_decoder_d1_layer_call_and_return_conditional_losses_96362$
"decoder_d1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_96572
activation_1/PartitionedCallÉ
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_9748decoder_output_9750*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_decoder_output_layer_call_and_return_conditional_losses_96762(
&decoder_output/StatefulPartitionedCallŇ
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ä
´
A__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_fn_9842

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*e
f`R^
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_97542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¸
b
F__inference_activation_1_layer_call_and_return_conditional_losses_9866

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˘

-__inference_decoder_output_layer_call_fn_9891

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_decoder_output_layer_call_and_return_conditional_losses_96762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

~
)__inference_decoder_d1_layer_call_fn_9861

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallň
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_decoder_d1_layer_call_and_return_conditional_losses_96362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

Á
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_9816

inputs-
)decoder_d1_matmul_readvariableop_resource.
*decoder_d1_biasadd_readvariableop_resource1
-decoder_output_matmul_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identityŻ
 decoder_d1/MatMul/ReadVariableOpReadVariableOp)decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 decoder_d1/MatMul/ReadVariableOp
decoder_d1/MatMulMatMulinputs(decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
decoder_d1/MatMulŽ
!decoder_d1/BiasAdd/ReadVariableOpReadVariableOp*decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!decoder_d1/BiasAdd/ReadVariableOpŽ
decoder_d1/BiasAddBiasAdddecoder_d1/MatMul:product:0)decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
decoder_d1/BiasAdd~
activation_1/ReluReludecoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
activation_1/Reluź
$decoder_output/MatMul/ReadVariableOpReadVariableOp-decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$decoder_output/MatMul/ReadVariableOpş
decoder_output/MatMulMatMulactivation_1/Relu:activations:0,decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
decoder_output/MatMulş
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%decoder_output/BiasAdd/ReadVariableOpž
decoder_output/BiasAddBiasAdddecoder_output/MatMul:product:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
decoder_output/BiasAdd
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
decoder_output/Sigmoido
IdentityIdentitydecoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙:::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

G
+__inference_activation_1_layer_call_fn_9871

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_96572
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*'
_input_shapes
:˙˙˙˙˙˙˙˙˙:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ô
°
H__inference_decoder_output_layer_call_and_return_conditional_losses_9882

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Sigmoid`
IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ż
Ü
 __inference__traced_restore_9954
file_prefix&
"assignvariableop_decoder_d1_kernel&
"assignvariableop_1_decoder_d1_bias,
(assignvariableop_2_decoder_output_kernel*
&assignvariableop_3_decoder_output_bias

identity_5˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2˘AssignVariableOp_3˘	RestoreV2˘RestoreV2_1ĺ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ń
valueçBäB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
RestoreV2/shape_and_slicesż
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp"assignvariableop_decoder_d1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp"assignvariableop_1_decoder_d1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp(assignvariableop_2_decoder_output_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp&assignvariableop_3_decoder_output_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpş

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4Ć

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Í
ˇ
A__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_fn_9737
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCall	latent_inunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*e
f`R^
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_97262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	latent_in:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¸ 
Ě
__inference__traced_save_9930
file_prefix0
,savev2_decoder_d1_kernel_read_readvariableop.
*savev2_decoder_d1_bias_read_readvariableop4
0savev2_decoder_output_kernel_read_readvariableop2
.savev2_decoder_output_bias_read_readvariableop
savev2_1_const

identity_1˘MergeV2Checkpoints˘SaveV2˘SaveV2_1
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
value3B1 B+_temp_442789ded92847aabd0aea3d1e980b06/part2	
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameß
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ń
valueçBäB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
SaveV2/shape_and_slicesë
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_decoder_d1_kernel_read_readvariableop*savev2_decoder_d1_bias_read_readvariableop0savev2_decoder_output_kernel_read_readvariableop.savev2_decoder_output_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardŹ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1˘
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesĎ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ă
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesŹ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
): :	::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: 
Í
ˇ
A__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_fn_9765
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCall	latent_inunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*e
f`R^
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_97542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	latent_in:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

Ź
D__inference_decoder_d1_layer_call_and_return_conditional_losses_9852

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
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ť
¨
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_9693
	latent_in
decoder_d1_9647
decoder_d1_9649
decoder_output_9687
decoder_output_9689
identity˘"decoder_d1/StatefulPartitionedCall˘&decoder_output/StatefulPartitionedCall
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCall	latent_indecoder_d1_9647decoder_d1_9649*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_decoder_d1_layer_call_and_return_conditional_losses_96362$
"decoder_d1/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_96572
activation_1/PartitionedCallÉ
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_9687decoder_output_9689*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_decoder_output_layer_call_and_return_conditional_losses_96762(
&decoder_output/StatefulPartitionedCallŇ
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:R N
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	latent_in:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ô
°
H__inference_decoder_output_layer_call_and_return_conditional_losses_9676

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Sigmoid`
IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

Á
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_9798

inputs-
)decoder_d1_matmul_readvariableop_resource.
*decoder_d1_biasadd_readvariableop_resource1
-decoder_output_matmul_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identityŻ
 decoder_d1/MatMul/ReadVariableOpReadVariableOp)decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 decoder_d1/MatMul/ReadVariableOp
decoder_d1/MatMulMatMulinputs(decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
decoder_d1/MatMulŽ
!decoder_d1/BiasAdd/ReadVariableOpReadVariableOp*decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!decoder_d1/BiasAdd/ReadVariableOpŽ
decoder_d1/BiasAddBiasAdddecoder_d1/MatMul:product:0)decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
decoder_d1/BiasAdd~
activation_1/ReluReludecoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
activation_1/Reluź
$decoder_output/MatMul/ReadVariableOpReadVariableOp-decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$decoder_output/MatMul/ReadVariableOpş
decoder_output/MatMulMatMulactivation_1/Relu:activations:0,decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
decoder_output/MatMulş
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%decoder_output/BiasAdd/ReadVariableOpž
decoder_output/BiasAddBiasAdddecoder_output/MatMul:product:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
decoder_output/BiasAdd
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
decoder_output/Sigmoido
IdentityIdentitydecoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙:::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "ŤL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ś
serving_default˘
?
	latent_in2
serving_default_latent_in:0˙˙˙˙˙˙˙˙˙C
decoder_output1
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:s

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
.__call__
/_default_save_signature
*0&call_and_return_all_conditional_losses"ű
_tf_keras_networkß{"class_name": "Functional", "name": "dec-ref-512-16-msle-g-relu-cifar10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dec-ref-512-16-msle-g-relu-cifar10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 3072, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "dec-ref-512-16-msle-g-relu-cifar10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 3072, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}}}
ď"ě
_tf_keras_input_layerĚ{"class_name": "InputLayer", "name": "latent_in", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}}
Ö


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
1__call__
*2&call_and_return_all_conditional_losses"ą
_tf_keras_layer{"class_name": "Dense", "name": "decoder_d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
˛
regularization_losses
trainable_variables
	variables
	keras_api
3__call__
*4&call_and_return_all_conditional_losses"Ł
_tf_keras_layer{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
â

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
5__call__
*6&call_and_return_all_conditional_losses"˝
_tf_keras_layerŁ{"class_name": "Dense", "name": "decoder_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 3072, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
 "
trackable_list_wrapper
<

0
1
2
3"
trackable_list_wrapper
<

0
1
2
3"
trackable_list_wrapper
Ę
metrics
non_trainable_variables
regularization_losses
layer_metrics
trainable_variables
	variables

layers
layer_regularization_losses
.__call__
/_default_save_signature
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
,
7serving_default"
signature_map
$:"	2decoder_d1/kernel
:2decoder_d1/bias
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
­
metrics
 non_trainable_variables
regularization_losses
!layer_metrics
trainable_variables
	variables

"layers
#layer_regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
$metrics
%non_trainable_variables
regularization_losses
&layer_metrics
trainable_variables
	variables

'layers
(layer_regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
):'
2decoder_output/kernel
": 2decoder_output/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
)metrics
*non_trainable_variables
regularization_losses
+layer_metrics
trainable_variables
	variables

,layers
-layer_regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
1
2
3"
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
Ň2Ď
A__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_fn_9737
A__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_fn_9842
A__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_fn_9765
A__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_fn_9829Ŕ
ˇ˛ł
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
kwonlydefaultsŞ 
annotationsŞ *
 
ß2Ü
__inference__wrapped_model_9622¸
˛
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
annotationsŞ *(˘%
# 
	latent_in˙˙˙˙˙˙˙˙˙
ž2ť
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_9798
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_9816
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_9693
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_9708Ŕ
ˇ˛ł
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
kwonlydefaultsŞ 
annotationsŞ *
 
Ó2Đ
)__inference_decoder_d1_layer_call_fn_9861˘
˛
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
annotationsŞ *
 
î2ë
D__inference_decoder_d1_layer_call_and_return_conditional_losses_9852˘
˛
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
annotationsŞ *
 
Ő2Ň
+__inference_activation_1_layer_call_fn_9871˘
˛
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
annotationsŞ *
 
đ2í
F__inference_activation_1_layer_call_and_return_conditional_losses_9866˘
˛
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
annotationsŞ *
 
×2Ô
-__inference_decoder_output_layer_call_fn_9891˘
˛
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
annotationsŞ *
 
ň2ď
H__inference_decoder_output_layer_call_and_return_conditional_losses_9882˘
˛
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
annotationsŞ *
 
3B1
"__inference_signature_wrapper_9780	latent_in
__inference__wrapped_model_9622|
2˘/
(˘%
# 
	latent_in˙˙˙˙˙˙˙˙˙
Ş "@Ş=
;
decoder_output)&
decoder_output˙˙˙˙˙˙˙˙˙¤
F__inference_activation_1_layer_call_and_return_conditional_losses_9866Z0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 |
+__inference_activation_1_layer_call_fn_9871M0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ę
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_9693j
:˘7
0˘-
# 
	latent_in˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 Ę
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_9708j
:˘7
0˘-
# 
	latent_in˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 Ç
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_9798g
7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 Ç
\__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_9816g
7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ˘
A__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_fn_9737]
:˘7
0˘-
# 
	latent_in˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙˘
A__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_fn_9765]
:˘7
0˘-
# 
	latent_in˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
A__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_fn_9829Z
7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
A__inference_dec-ref-512-16-msle-g-relu-cifar10_layer_call_fn_9842Z
7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙Ľ
D__inference_decoder_d1_layer_call_and_return_conditional_losses_9852]
/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 }
)__inference_decoder_d1_layer_call_fn_9861P
/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ş
H__inference_decoder_output_layer_call_and_return_conditional_losses_9882^0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
-__inference_decoder_output_layer_call_fn_9891Q0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙°
"__inference_signature_wrapper_9780
?˘<
˘ 
5Ş2
0
	latent_in# 
	latent_in˙˙˙˙˙˙˙˙˙"@Ş=
;
decoder_output)&
decoder_output˙˙˙˙˙˙˙˙˙