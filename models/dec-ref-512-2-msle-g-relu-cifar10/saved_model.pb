Ň°
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
 "serve*2.4.0-dev202007052v1.12.1-35711-g8202ae9d5c8ä´
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
	variables
trainable_variables
	keras_api
	
signatures
 
h


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
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
non_trainable_variables
regularization_losses

layers
metrics
layer_metrics
	variables
trainable_variables
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
non_trainable_variables
regularization_losses

 layers
!metrics
"layer_metrics
	variables
trainable_variables
#layer_regularization_losses
 
 
 
­
$non_trainable_variables
regularization_losses

%layers
&metrics
'layer_metrics
	variables
trainable_variables
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
)non_trainable_variables
regularization_losses

*layers
+metrics
,layer_metrics
	variables
trainable_variables
-layer_regularization_losses
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
 
 
|
serving_default_latent_inPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

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
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_15720
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Á
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
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_15866
ě
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
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_15888ę
š
c
G__inference_activation_1_layer_call_and_return_conditional_losses_15597

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
Ň
´
A__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_fn_15782

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallŞ
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
*0
config_proto 

CPU

GPU2*0J 8 *e
f`R^
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_156942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ó
­
E__inference_decoder_d1_layer_call_and_return_conditional_losses_15576

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
:˙˙˙˙˙˙˙˙˙:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Đ
Ź
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_15648
	latent_in
decoder_d1_15636
decoder_d1_15638
decoder_output_15642
decoder_output_15644
identity˘"decoder_d1/StatefulPartitionedCall˘&decoder_output/StatefulPartitionedCall˘
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCall	latent_indecoder_d1_15636decoder_d1_15638*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_155762$
"decoder_d1/StatefulPartitionedCall
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
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_155972
activation_1/PartitionedCallŇ
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_15642decoder_output_15644*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_156162(
&decoder_output/StatefulPartitionedCallŇ
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:R N
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	latent_in

Á
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_15738

inputs-
)decoder_d1_matmul_readvariableop_resource.
*decoder_d1_biasadd_readvariableop_resource1
-decoder_output_matmul_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identityŻ
 decoder_d1/MatMul/ReadVariableOpReadVariableOp)decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
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
#:˙˙˙˙˙˙˙˙˙:::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
š
ą
I__inference_decoder_output_layer_call_and_return_conditional_losses_15616

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
 
_user_specified_nameinputs
Đ
Ź
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_15633
	latent_in
decoder_d1_15587
decoder_d1_15589
decoder_output_15627
decoder_output_15629
identity˘"decoder_d1/StatefulPartitionedCall˘&decoder_output/StatefulPartitionedCall˘
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCall	latent_indecoder_d1_15587decoder_d1_15589*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_155762$
"decoder_d1/StatefulPartitionedCall
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
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_155972
activation_1/PartitionedCallŇ
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_15627decoder_output_15629*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_156162(
&decoder_output/StatefulPartitionedCallŇ
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:R N
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	latent_in
Ç
Š
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_15666

inputs
decoder_d1_15654
decoder_d1_15656
decoder_output_15660
decoder_output_15662
identity˘"decoder_d1/StatefulPartitionedCall˘&decoder_output/StatefulPartitionedCall
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_d1_15654decoder_d1_15656*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_155762$
"decoder_d1/StatefulPartitionedCall
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
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_155972
activation_1/PartitionedCallŇ
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_15660decoder_output_15662*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_156162(
&decoder_output/StatefulPartitionedCallŇ
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ň
´
A__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_fn_15769

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallŞ
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
*0
config_proto 

CPU

GPU2*0J 8 *e
f`R^
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_156662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
đ

 __inference__wrapped_model_15562
	latent_inO
Kdec_ref_512_2_msle_g_relu_cifar10_decoder_d1_matmul_readvariableop_resourceP
Ldec_ref_512_2_msle_g_relu_cifar10_decoder_d1_biasadd_readvariableop_resourceS
Odec_ref_512_2_msle_g_relu_cifar10_decoder_output_matmul_readvariableop_resourceT
Pdec_ref_512_2_msle_g_relu_cifar10_decoder_output_biasadd_readvariableop_resource
identity
Bdec-ref-512-2-msle-g-relu-cifar10/decoder_d1/MatMul/ReadVariableOpReadVariableOpKdec_ref_512_2_msle_g_relu_cifar10_decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02D
Bdec-ref-512-2-msle-g-relu-cifar10/decoder_d1/MatMul/ReadVariableOpţ
3dec-ref-512-2-msle-g-relu-cifar10/decoder_d1/MatMulMatMul	latent_inJdec-ref-512-2-msle-g-relu-cifar10/decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙25
3dec-ref-512-2-msle-g-relu-cifar10/decoder_d1/MatMul
Cdec-ref-512-2-msle-g-relu-cifar10/decoder_d1/BiasAdd/ReadVariableOpReadVariableOpLdec_ref_512_2_msle_g_relu_cifar10_decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02E
Cdec-ref-512-2-msle-g-relu-cifar10/decoder_d1/BiasAdd/ReadVariableOpś
4dec-ref-512-2-msle-g-relu-cifar10/decoder_d1/BiasAddBiasAdd=dec-ref-512-2-msle-g-relu-cifar10/decoder_d1/MatMul:product:0Kdec-ref-512-2-msle-g-relu-cifar10/decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙26
4dec-ref-512-2-msle-g-relu-cifar10/decoder_d1/BiasAddä
3dec-ref-512-2-msle-g-relu-cifar10/activation_1/ReluRelu=dec-ref-512-2-msle-g-relu-cifar10/decoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙25
3dec-ref-512-2-msle-g-relu-cifar10/activation_1/Relu˘
Fdec-ref-512-2-msle-g-relu-cifar10/decoder_output/MatMul/ReadVariableOpReadVariableOpOdec_ref_512_2_msle_g_relu_cifar10_decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02H
Fdec-ref-512-2-msle-g-relu-cifar10/decoder_output/MatMul/ReadVariableOpÂ
7dec-ref-512-2-msle-g-relu-cifar10/decoder_output/MatMulMatMulAdec-ref-512-2-msle-g-relu-cifar10/activation_1/Relu:activations:0Ndec-ref-512-2-msle-g-relu-cifar10/decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙29
7dec-ref-512-2-msle-g-relu-cifar10/decoder_output/MatMul 
Gdec-ref-512-2-msle-g-relu-cifar10/decoder_output/BiasAdd/ReadVariableOpReadVariableOpPdec_ref_512_2_msle_g_relu_cifar10_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02I
Gdec-ref-512-2-msle-g-relu-cifar10/decoder_output/BiasAdd/ReadVariableOpĆ
8dec-ref-512-2-msle-g-relu-cifar10/decoder_output/BiasAddBiasAddAdec-ref-512-2-msle-g-relu-cifar10/decoder_output/MatMul:product:0Odec-ref-512-2-msle-g-relu-cifar10/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2:
8dec-ref-512-2-msle-g-relu-cifar10/decoder_output/BiasAddő
8dec-ref-512-2-msle-g-relu-cifar10/decoder_output/SigmoidSigmoidAdec-ref-512-2-msle-g-relu-cifar10/decoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2:
8dec-ref-512-2-msle-g-relu-cifar10/decoder_output/Sigmoid
IdentityIdentity<dec-ref-512-2-msle-g-relu-cifar10/decoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙:::::R N
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	latent_in
Ű
ˇ
A__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_fn_15677
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall­
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
*0
config_proto 

CPU

GPU2*0J 8 *e
f`R^
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_156662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	latent_in
Ó
­
E__inference_decoder_d1_layer_call_and_return_conditional_losses_15792

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
:˙˙˙˙˙˙˙˙˙:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ă

*__inference_decoder_d1_layer_call_fn_15801

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallů
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
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_155762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

H
,__inference_activation_1_layer_call_fn_15811

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
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_155972
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
Ç
Š
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_15694

inputs
decoder_d1_15682
decoder_d1_15684
decoder_output_15688
decoder_output_15690
identity˘"decoder_d1/StatefulPartitionedCall˘&decoder_output/StatefulPartitionedCall
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_d1_15682decoder_d1_15684*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_d1_layer_call_and_return_conditional_losses_155762$
"decoder_d1/StatefulPartitionedCall
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
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_155972
activation_1/PartitionedCallŇ
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_15688decoder_output_15690*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_156162(
&decoder_output/StatefulPartitionedCallŇ
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¤
ˇ
__inference__traced_save_15866
file_prefix0
,savev2_decoder_d1_kernel_read_readvariableop.
*savev2_decoder_d1_bias_read_readvariableop4
0savev2_decoder_output_kernel_read_readvariableop2
.savev2_decoder_output_bias_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
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
value3B1 B+_temp_38edf3b34ede41869a1616cd9ce5d572/part2	
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
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameý
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slicesú
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_decoder_d1_kernel_read_readvariableop*savev2_decoder_d1_bias_read_readvariableop0savev2_decoder_output_kernel_read_readvariableop.savev2_decoder_output_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
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

identity_1Identity_1:output:0*<
_input_shapes+
): :	::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!
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
ä
Ă
!__inference__traced_restore_15888
file_prefix&
"assignvariableop_decoder_d1_kernel&
"assignvariableop_1_decoder_d1_bias,
(assignvariableop_2_decoder_output_kernel*
&assignvariableop_3_decoder_output_bias

identity_5˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2˘AssignVariableOp_3
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slicesÄ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityĄ
AssignVariableOpAssignVariableOp"assignvariableop_decoder_d1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_decoder_d1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2­
AssignVariableOp_2AssignVariableOp(assignvariableop_2_decoder_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ť
AssignVariableOp_3AssignVariableOp&assignvariableop_3_decoder_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpş

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4Ź

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
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
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
î

.__inference_decoder_output_layer_call_fn_15831

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallý
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
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_156162
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
 
_user_specified_nameinputs


#__inference_signature_wrapper_15720
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCallń
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
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_155622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	latent_in
š
ą
I__inference_decoder_output_layer_call_and_return_conditional_losses_15822

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
 
_user_specified_nameinputs

Á
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_15756

inputs-
)decoder_d1_matmul_readvariableop_resource.
*decoder_d1_biasadd_readvariableop_resource1
-decoder_output_matmul_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identityŻ
 decoder_d1/MatMul/ReadVariableOpReadVariableOp)decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	*
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
#:˙˙˙˙˙˙˙˙˙:::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ű
ˇ
A__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_fn_15705
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall­
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
*0
config_proto 

CPU

GPU2*0J 8 *e
f`R^
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_156942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#
_user_specified_name	latent_in
š
c
G__inference_activation_1_layer_call_and_return_conditional_losses_15806

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
 
_user_specified_nameinputs"ÄL
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
serving_default_latent_in:0˙˙˙˙˙˙˙˙˙C
decoder_output1
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:t
š
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
._default_save_signature
*/&call_and_return_all_conditional_losses
0__call__"
_tf_keras_networků{"class_name": "Functional", "name": "dec-ref-512-2-msle-g-relu-cifar10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "dec-ref-512-2-msle-g-relu-cifar10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 3072, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "dec-ref-512-2-msle-g-relu-cifar10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 3072, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}}}
í"ę
_tf_keras_input_layerĘ{"class_name": "InputLayer", "name": "latent_in", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}}
÷


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*1&call_and_return_all_conditional_losses
2__call__"Ň
_tf_keras_layer¸{"class_name": "Dense", "name": "decoder_d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
Ő
regularization_losses
	variables
trainable_variables
	keras_api
*3&call_and_return_all_conditional_losses
4__call__"Ć
_tf_keras_layerŹ{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*5&call_and_return_all_conditional_losses
6__call__"ŕ
_tf_keras_layerĆ{"class_name": "Dense", "name": "decoder_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 3072, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
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
non_trainable_variables
regularization_losses

layers
metrics
layer_metrics
	variables
trainable_variables
layer_regularization_losses
0__call__
._default_save_signature
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
,
7serving_default"
signature_map
$:"	2decoder_d1/kernel
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
non_trainable_variables
regularization_losses

 layers
!metrics
"layer_metrics
	variables
trainable_variables
#layer_regularization_losses
2__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
$non_trainable_variables
regularization_losses

%layers
&metrics
'layer_metrics
	variables
trainable_variables
(layer_regularization_losses
4__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
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
)non_trainable_variables
regularization_losses

*layers
+metrics
,layer_metrics
	variables
trainable_variables
-layer_regularization_losses
6__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
3"
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
ŕ2Ý
 __inference__wrapped_model_15562¸
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
	latent_in˙˙˙˙˙˙˙˙˙
ž2ť
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_15648
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_15756
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_15633
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_15738Ŕ
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
Ň2Ď
A__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_fn_15782
A__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_fn_15769
A__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_fn_15705
A__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_fn_15677Ŕ
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
ď2ě
E__inference_decoder_d1_layer_call_and_return_conditional_losses_15792˘
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
Ô2Ń
*__inference_decoder_d1_layer_call_fn_15801˘
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
ń2î
G__inference_activation_1_layer_call_and_return_conditional_losses_15806˘
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
Ö2Ó
,__inference_activation_1_layer_call_fn_15811˘
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
ó2đ
I__inference_decoder_output_layer_call_and_return_conditional_losses_15822˘
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
Ř2Ő
.__inference_decoder_output_layer_call_fn_15831˘
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
4B2
#__inference_signature_wrapper_15720	latent_in 
 __inference__wrapped_model_15562|
2˘/
(˘%
# 
	latent_in˙˙˙˙˙˙˙˙˙
Ş "@Ş=
;
decoder_output)&
decoder_output˙˙˙˙˙˙˙˙˙Ľ
G__inference_activation_1_layer_call_and_return_conditional_losses_15806Z0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 }
,__inference_activation_1_layer_call_fn_15811M0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ę
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_15633j
:˘7
0˘-
# 
	latent_in˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 Ę
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_15648j
:˘7
0˘-
# 
	latent_in˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 Ç
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_15738g
7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 Ç
\__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_and_return_conditional_losses_15756g
7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ˘
A__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_fn_15677]
:˘7
0˘-
# 
	latent_in˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙˘
A__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_fn_15705]
:˘7
0˘-
# 
	latent_in˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
A__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_fn_15769Z
7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
A__inference_dec-ref-512-2-msle-g-relu-cifar10_layer_call_fn_15782Z
7˘4
-˘*
 
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙Ś
E__inference_decoder_d1_layer_call_and_return_conditional_losses_15792]
/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 ~
*__inference_decoder_d1_layer_call_fn_15801P
/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ť
I__inference_decoder_output_layer_call_and_return_conditional_losses_15822^0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
.__inference_decoder_output_layer_call_fn_15831Q0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙ą
#__inference_signature_wrapper_15720
?˘<
˘ 
5Ş2
0
	latent_in# 
	latent_in˙˙˙˙˙˙˙˙˙"@Ş=
;
decoder_output)&
decoder_output˙˙˙˙˙˙˙˙˙