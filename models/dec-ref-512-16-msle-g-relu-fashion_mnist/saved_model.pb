К┤
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
 ѕ"serve*2.4.0-dev202007052v1.12.1-35711-g8202ae9d5c8ли

decoder_d1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*"
shared_namedecoder_d1/kernel
x
%decoder_d1/kernel/Read/ReadVariableOpReadVariableOpdecoder_d1/kernel*
_output_shapes
:	ђ*
dtype0
w
decoder_d1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ* 
shared_namedecoder_d1/bias
p
#decoder_d1/bias/Read/ReadVariableOpReadVariableOpdecoder_d1/bias*
_output_shapes	
:ђ*
dtype0
ѕ
decoder_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђљ*&
shared_namedecoder_output/kernel
Ђ
)decoder_output/kernel/Read/ReadVariableOpReadVariableOpdecoder_output/kernel* 
_output_shapes
:
ђљ*
dtype0

decoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:љ*$
shared_namedecoder_output/bias
x
'decoder_output/bias/Read/ReadVariableOpReadVariableOpdecoder_output/bias*
_output_shapes	
:љ*
dtype0

NoOpNoOp
┌
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ћ
valueІBѕ BЂ
╩
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
 
h


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api


0
1
2
3
 


0
1
2
3
Г
layer_regularization_losses

layers
trainable_variables
layer_metrics
non_trainable_variables
regularization_losses
metrics
	variables
 
][
VARIABLE_VALUEdecoder_d1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdecoder_d1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1
 


0
1
Г
layer_regularization_losses

 layers
!layer_metrics
trainable_variables
"non_trainable_variables
regularization_losses
#metrics
	variables
 
 
 
Г
$layer_regularization_losses

%layers
&layer_metrics
trainable_variables
'non_trainable_variables
regularization_losses
(metrics
	variables
a_
VARIABLE_VALUEdecoder_output/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEdecoder_output/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
)layer_regularization_losses

*layers
+layer_metrics
trainable_variables
,non_trainable_variables
regularization_losses
-metrics
	variables
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
:         *
dtype0*
shape:         
Ј
StatefulPartitionedCallStatefulPartitionedCallserving_default_latent_indecoder_d1/kerneldecoder_d1/biasdecoder_output/kerneldecoder_output/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *,
f'R%
#__inference_signature_wrapper_17260
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
┴
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
GPU2*0J 8ѓ *'
f"R 
__inference__traced_save_17406
В
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
GPU2*0J 8ѓ **
f%R#
!__inference__traced_restore_17428оЋ
Я
╗
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_17309

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *l
fgRe
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_172062
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ж
Й
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_17245
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCall	latent_inunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *l
fgRe
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_172342
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:         
#
_user_specified_name	latent_in
╬
░
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_17234

inputs
decoder_d1_17222
decoder_d1_17224
decoder_output_17228
decoder_output_17230
identityѕб"decoder_d1/StatefulPartitionedCallб&decoder_output/StatefulPartitionedCallЪ
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_d1_17222decoder_d1_17224*
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
E__inference_decoder_d1_layer_call_and_return_conditional_losses_171162$
"decoder_d1/StatefulPartitionedCallѕ
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_171372
activation_1/PartitionedCallм
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_17228decoder_output_17230*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_171562(
&decoder_output/StatefulPartitionedCallм
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
М
Г
E__inference_decoder_d1_layer_call_and_return_conditional_losses_17332

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
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
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Щ
г
 __inference__wrapped_model_17102
	latent_inV
Rdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_matmul_readvariableop_resourceW
Sdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_biasadd_readvariableop_resourceZ
Vdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_matmul_readvariableop_resource[
Wdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_biasadd_readvariableop_resource
identityѕф
Idec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOpReadVariableOpRdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02K
Idec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOpЊ
:dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMulMatMul	latent_inQdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2<
:dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMulЕ
Jdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOpReadVariableOpSdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02L
Jdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOpм
;dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAddBiasAddDdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/MatMul:product:0Rdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2=
;dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAddщ
:dec-ref-512-16-msle-g-relu-fashion_mnist/activation_1/ReluReluDdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2<
:dec-ref-512-16-msle-g-relu-fashion_mnist/activation_1/Reluи
Mdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOpReadVariableOpVdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
ђљ*
dtype02O
Mdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOpя
>dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMulMatMulHdec-ref-512-16-msle-g-relu-fashion_mnist/activation_1/Relu:activations:0Udec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ2@
>dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMulх
Ndec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOpReadVariableOpWdec_ref_512_16_msle_g_relu_fashion_mnist_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype02P
Ndec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOpР
?dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAddBiasAddHdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/MatMul:product:0Vdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ2A
?dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAddі
?dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/SigmoidSigmoidHdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:         љ2A
?dec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/Sigmoidў
IdentityIdentityCdec-ref-512-16-msle-g-relu-fashion_mnist/decoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         :::::R N
'
_output_shapes
:         
#
_user_specified_name	latent_in
╣
c
G__inference_activation_1_layer_call_and_return_conditional_losses_17346

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
О
│
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_17188
	latent_in
decoder_d1_17176
decoder_d1_17178
decoder_output_17182
decoder_output_17184
identityѕб"decoder_d1/StatefulPartitionedCallб&decoder_output/StatefulPartitionedCallб
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCall	latent_indecoder_d1_17176decoder_d1_17178*
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
E__inference_decoder_d1_layer_call_and_return_conditional_losses_171162$
"decoder_d1/StatefulPartitionedCallѕ
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_171372
activation_1/PartitionedCallм
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_17182decoder_output_17184*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_171562(
&decoder_output/StatefulPartitionedCallм
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:R N
'
_output_shapes
:         
#
_user_specified_name	latent_in
╣
c
G__inference_activation_1_layer_call_and_return_conditional_losses_17137

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
Ъ
H
,__inference_activation_1_layer_call_fn_17351

inputs
identity╔
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
GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_171372
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
ю
╚
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_17296

inputs-
)decoder_d1_matmul_readvariableop_resource.
*decoder_d1_biasadd_readvariableop_resource1
-decoder_output_matmul_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identityѕ»
 decoder_d1/MatMul/ReadVariableOpReadVariableOp)decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02"
 decoder_d1/MatMul/ReadVariableOpЋ
decoder_d1/MatMulMatMulinputs(decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
decoder_d1/MatMul«
!decoder_d1/BiasAdd/ReadVariableOpReadVariableOp*decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02#
!decoder_d1/BiasAdd/ReadVariableOp«
decoder_d1/BiasAddBiasAdddecoder_d1/MatMul:product:0)decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
decoder_d1/BiasAdd~
activation_1/ReluReludecoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
activation_1/Relu╝
$decoder_output/MatMul/ReadVariableOpReadVariableOp-decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
ђљ*
dtype02&
$decoder_output/MatMul/ReadVariableOp║
decoder_output/MatMulMatMulactivation_1/Relu:activations:0,decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ2
decoder_output/MatMul║
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype02'
%decoder_output/BiasAdd/ReadVariableOpЙ
decoder_output/BiasAddBiasAdddecoder_output/MatMul:product:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ2
decoder_output/BiasAddЈ
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:         љ2
decoder_output/Sigmoido
IdentityIdentitydecoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         :::::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╣
▒
I__inference_decoder_output_layer_call_and_return_conditional_losses_17156

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђљ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         љ2	
Sigmoid`
IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:::P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
М
Г
E__inference_decoder_d1_layer_call_and_return_conditional_losses_17116

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
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
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ђ
Ў
#__inference_signature_wrapper_17260
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCall	latent_inunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *)
f$R"
 __inference__wrapped_model_171022
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:         
#
_user_specified_name	latent_in
с

*__inference_decoder_d1_layer_call_fn_17341

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
E__inference_decoder_d1_layer_call_and_return_conditional_losses_171162
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╣
▒
I__inference_decoder_output_layer_call_and_return_conditional_losses_17362

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђљ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         љ2	
Sigmoid`
IdentityIdentitySigmoid:y:0*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:::P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
О
│
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_17173
	latent_in
decoder_d1_17127
decoder_d1_17129
decoder_output_17167
decoder_output_17169
identityѕб"decoder_d1/StatefulPartitionedCallб&decoder_output/StatefulPartitionedCallб
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCall	latent_indecoder_d1_17127decoder_d1_17129*
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
E__inference_decoder_d1_layer_call_and_return_conditional_losses_171162$
"decoder_d1/StatefulPartitionedCallѕ
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_171372
activation_1/PartitionedCallм
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_17167decoder_output_17169*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_171562(
&decoder_output/StatefulPartitionedCallм
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:R N
'
_output_shapes
:         
#
_user_specified_name	latent_in
Ь
Ѓ
.__inference_decoder_output_layer_call_fn_17371

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_171562
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
С
├
!__inference__traced_restore_17428
file_prefix&
"assignvariableop_decoder_d1_kernel&
"assignvariableop_1_decoder_d1_bias,
(assignvariableop_2_decoder_output_kernel*
&assignvariableop_3_decoder_output_bias

identity_5ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3Ѓ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ј
valueЁBѓB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesў
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices─
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

IdentityА
AssignVariableOpAssignVariableOp"assignvariableop_decoder_d1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Д
AssignVariableOp_1AssignVariableOp"assignvariableop_1_decoder_d1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Г
AssignVariableOp_2AssignVariableOp(assignvariableop_2_decoder_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ф
AssignVariableOp_3AssignVariableOp&assignvariableop_3_decoder_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp║

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4г

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
Я
╗
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_17322

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *l
fgRe
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_172342
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ю
╚
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_17278

inputs-
)decoder_d1_matmul_readvariableop_resource.
*decoder_d1_biasadd_readvariableop_resource1
-decoder_output_matmul_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identityѕ»
 decoder_d1/MatMul/ReadVariableOpReadVariableOp)decoder_d1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02"
 decoder_d1/MatMul/ReadVariableOpЋ
decoder_d1/MatMulMatMulinputs(decoder_d1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
decoder_d1/MatMul«
!decoder_d1/BiasAdd/ReadVariableOpReadVariableOp*decoder_d1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02#
!decoder_d1/BiasAdd/ReadVariableOp«
decoder_d1/BiasAddBiasAdddecoder_d1/MatMul:product:0)decoder_d1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
decoder_d1/BiasAdd~
activation_1/ReluReludecoder_d1/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
activation_1/Relu╝
$decoder_output/MatMul/ReadVariableOpReadVariableOp-decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
ђљ*
dtype02&
$decoder_output/MatMul/ReadVariableOp║
decoder_output/MatMulMatMulactivation_1/Relu:activations:0,decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ2
decoder_output/MatMul║
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:љ*
dtype02'
%decoder_output/BiasAdd/ReadVariableOpЙ
decoder_output/BiasAddBiasAdddecoder_output/MatMul:product:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         љ2
decoder_output/BiasAddЈ
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:         љ2
decoder_output/Sigmoido
IdentityIdentitydecoder_output/Sigmoid:y:0*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         :::::O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ж
Й
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_17217
	latent_in
unknown
	unknown_0
	unknown_1
	unknown_2
identityѕбStatefulPartitionedCall┤
StatefulPartitionedCallStatefulPartitionedCall	latent_inunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *l
fgRe
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_172062
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:         
#
_user_specified_name	latent_in
╬
░
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_17206

inputs
decoder_d1_17194
decoder_d1_17196
decoder_output_17200
decoder_output_17202
identityѕб"decoder_d1/StatefulPartitionedCallб&decoder_output/StatefulPartitionedCallЪ
"decoder_d1/StatefulPartitionedCallStatefulPartitionedCallinputsdecoder_d1_17194decoder_d1_17196*
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
E__inference_decoder_d1_layer_call_and_return_conditional_losses_171162$
"decoder_d1/StatefulPartitionedCallѕ
activation_1/PartitionedCallPartitionedCall+decoder_d1/StatefulPartitionedCall:output:0*
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
GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_171372
activation_1/PartitionedCallм
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0decoder_output_17200decoder_output_17202*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_171562(
&decoder_output/StatefulPartitionedCallм
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0#^decoder_d1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall*
T0*(
_output_shapes
:         љ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::::2H
"decoder_d1/StatefulPartitionedCall"decoder_d1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ц
и
__inference__traced_save_17406
file_prefix0
,savev2_decoder_d1_kernel_read_readvariableop.
*savev2_decoder_d1_bias_read_readvariableop4
0savev2_decoder_output_kernel_read_readvariableop2
.savev2_decoder_output_bias_read_readvariableop
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
value3B1 B+_temp_7dac246d34d8427091b7b17241a13deb/part2	
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
ShardedFilename§
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ј
valueЁBѓB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesњ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slicesЩ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_decoder_d1_kernel_read_readvariableop*savev2_decoder_d1_bias_read_readvariableop0savev2_decoder_output_kernel_read_readvariableop.savev2_decoder_output_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
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

identity_1Identity_1:output:0*<
_input_shapes+
): :	ђ:ђ:
ђљ:љ: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	ђ:!

_output_shapes	
:ђ:&"
 
_output_shapes
:
ђљ:!

_output_shapes	
:љ:

_output_shapes
: "─L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Х
serving_defaultб
?
	latent_in2
serving_default_latent_in:0         C
decoder_output1
StatefulPartitionedCall:0         љtensorflow/serving/predict:еu
¤
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
.__call__
*/&call_and_return_all_conditional_losses
0_default_save_signature"Ф
_tf_keras_networkЈ{"class_name": "Functional", "name": "dec-ref-512-16-msle-g-relu-fashion_mnist", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "dec-ref-512-16-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "dec-ref-512-16-msle-g-relu-fashion_mnist", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}, "name": "latent_in", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_d1", "inbound_nodes": [[["latent_in", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["decoder_d1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoder_output", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}], "input_layers": [["latent_in", 0, 0]], "output_layers": [["decoder_output", 0, 0]]}}}
№"В
_tf_keras_input_layer╠{"class_name": "InputLayer", "name": "latent_in", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_in"}}
щ


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
1__call__
*2&call_and_return_all_conditional_losses"н
_tf_keras_layer║{"class_name": "Dense", "name": "decoder_d1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoder_d1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
Н
trainable_variables
regularization_losses
	variables
	keras_api
3__call__
*4&call_and_return_all_conditional_losses"к
_tf_keras_layerг{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
ё

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
5__call__
*6&call_and_return_all_conditional_losses"▀
_tf_keras_layer┼{"class_name": "Dense", "name": "decoder_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 784, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
<

0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<

0
1
2
3"
trackable_list_wrapper
╩
layer_regularization_losses

layers
trainable_variables
layer_metrics
non_trainable_variables
regularization_losses
metrics
	variables
.__call__
0_default_save_signature
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
,
7serving_default"
signature_map
$:"	ђ2decoder_d1/kernel
:ђ2decoder_d1/bias
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
Г
layer_regularization_losses

 layers
!layer_metrics
trainable_variables
"non_trainable_variables
regularization_losses
#metrics
	variables
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
Г
$layer_regularization_losses

%layers
&layer_metrics
trainable_variables
'non_trainable_variables
regularization_losses
(metrics
	variables
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
):'
ђљ2decoder_output/kernel
": љ2decoder_output/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Г
)layer_regularization_losses

*layers
+layer_metrics
trainable_variables
,non_trainable_variables
regularization_losses
-metrics
	variables
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
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
Ь2в
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_17245
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_17309
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_17217
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_17322└
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
┌2О
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_17173
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_17188
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_17296
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_17278└
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
Я2П
 __inference__wrapped_model_17102И
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
annotationsф *(б%
#і 
	latent_in         
н2Л
*__inference_decoder_d1_layer_call_fn_17341б
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
E__inference_decoder_d1_layer_call_and_return_conditional_losses_17332б
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
о2М
,__inference_activation_1_layer_call_fn_17351б
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
ы2Ь
G__inference_activation_1_layer_call_and_return_conditional_losses_17346б
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
п2Н
.__inference_decoder_output_layer_call_fn_17371б
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
з2­
I__inference_decoder_output_layer_call_and_return_conditional_losses_17362б
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
4B2
#__inference_signature_wrapper_17260	latent_inа
 __inference__wrapped_model_17102|
2б/
(б%
#і 
	latent_in         
ф "@ф=
;
decoder_output)і&
decoder_output         љЦ
G__inference_activation_1_layer_call_and_return_conditional_losses_17346Z0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ }
,__inference_activation_1_layer_call_fn_17351M0б-
&б#
!і
inputs         ђ
ф "і         ђЛ
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_17173j
:б7
0б-
#і 
	latent_in         
p

 
ф "&б#
і
0         љ
џ Л
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_17188j
:б7
0б-
#і 
	latent_in         
p 

 
ф "&б#
і
0         љ
џ ╬
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_17278g
7б4
-б*
 і
inputs         
p

 
ф "&б#
і
0         љ
џ ╬
c__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_and_return_conditional_losses_17296g
7б4
-б*
 і
inputs         
p 

 
ф "&б#
і
0         љ
џ Е
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_17217]
:б7
0б-
#і 
	latent_in         
p

 
ф "і         љЕ
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_17245]
:б7
0б-
#і 
	latent_in         
p 

 
ф "і         љд
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_17309Z
7б4
-б*
 і
inputs         
p

 
ф "і         љд
H__inference_dec-ref-512-16-msle-g-relu-fashion_mnist_layer_call_fn_17322Z
7б4
-б*
 і
inputs         
p 

 
ф "і         љд
E__inference_decoder_d1_layer_call_and_return_conditional_losses_17332]
/б,
%б"
 і
inputs         
ф "&б#
і
0         ђ
џ ~
*__inference_decoder_d1_layer_call_fn_17341P
/б,
%б"
 і
inputs         
ф "і         ђФ
I__inference_decoder_output_layer_call_and_return_conditional_losses_17362^0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         љ
џ Ѓ
.__inference_decoder_output_layer_call_fn_17371Q0б-
&б#
!і
inputs         ђ
ф "і         љ▒
#__inference_signature_wrapper_17260Ѕ
?б<
б 
5ф2
0
	latent_in#і 
	latent_in         "@ф=
;
decoder_output)і&
decoder_output         љ