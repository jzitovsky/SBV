
À
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
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
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ÙØ

Online/Conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameOnline/Conv/kernel

&Online/Conv/kernel/Read/ReadVariableOpReadVariableOpOnline/Conv/kernel*&
_output_shapes
: *
dtype0
x
Online/Conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameOnline/Conv/bias
q
$Online/Conv/bias/Read/ReadVariableOpReadVariableOpOnline/Conv/bias*
_output_shapes
: *
dtype0

Online/Conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*%
shared_nameOnline/Conv_1/kernel

(Online/Conv_1/kernel/Read/ReadVariableOpReadVariableOpOnline/Conv_1/kernel*&
_output_shapes
: @*
dtype0
|
Online/Conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameOnline/Conv_1/bias
u
&Online/Conv_1/bias/Read/ReadVariableOpReadVariableOpOnline/Conv_1/bias*
_output_shapes
:@*
dtype0

Online/Conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*%
shared_nameOnline/Conv_2/kernel

(Online/Conv_2/kernel/Read/ReadVariableOpReadVariableOpOnline/Conv_2/kernel*&
_output_shapes
:@@*
dtype0
|
Online/Conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameOnline/Conv_2/bias
u
&Online/Conv_2/bias/Read/ReadVariableOpReadVariableOpOnline/Conv_2/bias*
_output_shapes
:@*
dtype0

Online/fully_connected/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
À<*.
shared_nameOnline/fully_connected/kernel

1Online/fully_connected/kernel/Read/ReadVariableOpReadVariableOpOnline/fully_connected/kernel* 
_output_shapes
:
À<*
dtype0

Online/fully_connected/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameOnline/fully_connected/bias

/Online/fully_connected/bias/Read/ReadVariableOpReadVariableOpOnline/fully_connected/bias*
_output_shapes	
:*
dtype0

Online/fully_connected_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*0
shared_name!Online/fully_connected_1/kernel

3Online/fully_connected_1/kernel/Read/ReadVariableOpReadVariableOpOnline/fully_connected_1/kernel*
_output_shapes
:		*
dtype0

Online/fully_connected_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*.
shared_nameOnline/fully_connected_1/bias

1Online/fully_connected_1/bias/Read/ReadVariableOpReadVariableOpOnline/fully_connected_1/bias*
_output_shapes
:	*
dtype0

NoOpNoOp
À
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*û
valueñBî Bç
¨
	conv1
	conv2
	conv3
flatten

dense1

dense2
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
 trainable_variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
 
F
0
1
2
3
4
5
"6
#7
(8
)9
F
0
1
2
3
4
5
"6
#7
(8
)9
­
.metrics
/layer_regularization_losses
0non_trainable_variables
1layer_metrics
regularization_losses

2layers
	variables
	trainable_variables
 
OM
VARIABLE_VALUEOnline/Conv/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEOnline/Conv/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
3metrics
4layer_regularization_losses
5non_trainable_variables
6layer_metrics
regularization_losses

7layers
	variables
trainable_variables
QO
VARIABLE_VALUEOnline/Conv_1/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEOnline/Conv_1/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
8metrics
9layer_regularization_losses
:non_trainable_variables
;layer_metrics
regularization_losses

<layers
	variables
trainable_variables
QO
VARIABLE_VALUEOnline/Conv_2/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEOnline/Conv_2/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
=metrics
>layer_regularization_losses
?non_trainable_variables
@layer_metrics
regularization_losses

Alayers
	variables
trainable_variables
 
 
 
­
Bmetrics
Clayer_regularization_losses
Dnon_trainable_variables
Elayer_metrics
regularization_losses

Flayers
	variables
 trainable_variables
[Y
VARIABLE_VALUEOnline/fully_connected/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEOnline/fully_connected/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
­
Gmetrics
Hlayer_regularization_losses
Inon_trainable_variables
Jlayer_metrics
$regularization_losses

Klayers
%	variables
&trainable_variables
][
VARIABLE_VALUEOnline/fully_connected_1/kernel(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEOnline/fully_connected_1/bias&dense2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
­
Lmetrics
Mlayer_regularization_losses
Nnon_trainable_variables
Olayer_metrics
*regularization_losses

Players
+	variables
,trainable_variables
 
 
 
 
*
0
1
2
3
4
5
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

serving_default_input_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿTT
¼
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Online/Conv/kernelOnline/Conv/biasOnline/Conv_1/kernelOnline/Conv_1/biasOnline/Conv_2/kernelOnline/Conv_2/biasOnline/fully_connected/kernelOnline/fully_connected/biasOnline/fully_connected_1/kernelOnline/fully_connected_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference_signature_wrapper_965
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ä
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&Online/Conv/kernel/Read/ReadVariableOp$Online/Conv/bias/Read/ReadVariableOp(Online/Conv_1/kernel/Read/ReadVariableOp&Online/Conv_1/bias/Read/ReadVariableOp(Online/Conv_2/kernel/Read/ReadVariableOp&Online/Conv_2/bias/Read/ReadVariableOp1Online/fully_connected/kernel/Read/ReadVariableOp/Online/fully_connected/bias/Read/ReadVariableOp3Online/fully_connected_1/kernel/Read/ReadVariableOp1Online/fully_connected_1/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8 *&
f!R
__inference__traced_save_1118

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameOnline/Conv/kernelOnline/Conv/biasOnline/Conv_1/kernelOnline/Conv_1/biasOnline/Conv_2/kernelOnline/Conv_2/biasOnline/fully_connected/kernelOnline/fully_connected/biasOnline/fully_connected_1/kernelOnline/fully_connected_1/bias*
Tin
2*
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
GPU2*0J 8 *)
f$R"
 __inference__traced_restore_1158§


§
I__inference_fully_connected_layer_call_and_return_conditional_losses_1058

inputsH
5matmul_readvariableop_online_fully_connected_1_kernel:		B
4biasadd_readvariableop_online_fully_connected_1_bias:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¥
MatMul/ReadVariableOpReadVariableOp5matmul_readvariableop_online_fully_connected_1_kernel*
_output_shapes
:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
MatMul¡
BiasAdd/ReadVariableOpReadVariableOp4biasadd_readvariableop_online_fully_connected_1_bias*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ.

 __inference__traced_restore_1158
file_prefix=
#assignvariableop_online_conv_kernel: 1
#assignvariableop_1_online_conv_bias: A
'assignvariableop_2_online_conv_1_kernel: @3
%assignvariableop_3_online_conv_1_bias:@A
'assignvariableop_4_online_conv_2_kernel:@@3
%assignvariableop_5_online_conv_2_bias:@D
0assignvariableop_6_online_fully_connected_kernel:
À<=
.assignvariableop_7_online_fully_connected_bias:	E
2assignvariableop_8_online_fully_connected_1_kernel:		>
0assignvariableop_9_online_fully_connected_1_bias:	
identity_11¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9»
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ç
value½BºB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slicesâ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¢
AssignVariableOpAssignVariableOp#assignvariableop_online_conv_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¨
AssignVariableOp_1AssignVariableOp#assignvariableop_1_online_conv_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¬
AssignVariableOp_2AssignVariableOp'assignvariableop_2_online_conv_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ª
AssignVariableOp_3AssignVariableOp%assignvariableop_3_online_conv_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¬
AssignVariableOp_4AssignVariableOp'assignvariableop_4_online_conv_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ª
AssignVariableOp_5AssignVariableOp%assignvariableop_5_online_conv_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6µ
AssignVariableOp_6AssignVariableOp0assignvariableop_6_online_fully_connected_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7³
AssignVariableOp_7AssignVariableOp.assignvariableop_7_online_fully_connected_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8·
AssignVariableOp_8AssignVariableOp2assignvariableop_8_online_fully_connected_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9µ
AssignVariableOp_9AssignVariableOp0assignvariableop_9_online_fully_connected_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpº
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10­
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
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
»
®
#__inference_Conv_layer_call_fn_1019

inputs.
online_conv_2_kernel:@@ 
online_conv_2_bias:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsonline_conv_2_kernelonline_conv_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_Conv_layer_call_and_return_conditional_losses_7612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¤
H__inference_fully_connected_layer_call_and_return_conditional_losses_784

inputsG
3matmul_readvariableop_online_fully_connected_kernel:
À<A
2biasadd_readvariableop_online_fully_connected_bias:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¤
MatMul/ReadVariableOpReadVariableOp3matmul_readvariableop_online_fully_connected_kernel* 
_output_shapes
:
À<*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul 
BiasAdd/ReadVariableOpReadVariableOp2biasadd_readvariableop_online_fully_connected_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<
 
_user_specified_nameinputs


=__inference_Conv_layer_call_and_return_conditional_losses_994

inputsD
*conv2d_readvariableop_online_conv_1_kernel: @7
)biasadd_readvariableop_online_conv_1_bias:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¡
Conv2D/ReadVariableOpReadVariableOp*conv2d_readvariableop_online_conv_1_kernel*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp)biasadd_readvariableop_online_conv_1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¯%
²
?__inference_Online_layer_call_and_return_conditional_losses_803
input_11
conv_online_conv_kernel: #
conv_online_conv_bias: 5
conv_1_online_conv_1_kernel: @'
conv_1_online_conv_1_bias:@5
conv_2_online_conv_2_kernel:@@'
conv_2_online_conv_2_bias:@A
-fully_connected_online_fully_connected_kernel:
À<:
+fully_connected_online_fully_connected_bias:	D
1fully_connected_1_online_fully_connected_1_kernel:		=
/fully_connected_1_online_fully_connected_1_bias:	
identity¢Conv/StatefulPartitionedCall¢Conv_1/StatefulPartitionedCall¢Conv_2/StatefulPartitionedCall¢'fully_connected/StatefulPartitionedCall¢)fully_connected_1/StatefulPartitionedCallf
CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT2
Cast[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/yu
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT2	
truediv£
Conv/StatefulPartitionedCallStatefulPartitionedCalltruediv:z:0conv_online_conv_kernelconv_online_conv_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_Conv_layer_call_and_return_conditional_losses_7312
Conv/StatefulPartitionedCallÉ
Conv_1/StatefulPartitionedCallStatefulPartitionedCall%Conv/StatefulPartitionedCall:output:0conv_1_online_conv_1_kernelconv_1_online_conv_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_Conv_layer_call_and_return_conditional_losses_7462 
Conv_1/StatefulPartitionedCallË
Conv_2/StatefulPartitionedCallStatefulPartitionedCall'Conv_1/StatefulPartitionedCall:output:0conv_2_online_conv_2_kernelconv_2_online_conv_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_Conv_layer_call_and_return_conditional_losses_7612 
Conv_2/StatefulPartitionedCalló
flatten/PartitionedCallPartitionedCall'Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_7712
flatten/PartitionedCallþ
'fully_connected/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0-fully_connected_online_fully_connected_kernel+fully_connected_online_fully_connected_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_fully_connected_layer_call_and_return_conditional_losses_7842)
'fully_connected/StatefulPartitionedCall
)fully_connected_1/StatefulPartitionedCallStatefulPartitionedCall0fully_connected/StatefulPartitionedCall:output:01fully_connected_1_online_fully_connected_1_kernel/fully_connected_1_online_fully_connected_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_fully_connected_layer_call_and_return_conditional_losses_7982+
)fully_connected_1/StatefulPartitionedCall½
IdentityIdentity2fully_connected_1/StatefulPartitionedCall:output:0^Conv/StatefulPartitionedCall^Conv_1/StatefulPartitionedCall^Conv_2/StatefulPartitionedCall(^fully_connected/StatefulPartitionedCall*^fully_connected_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿTT: : : : : : : : : : 2<
Conv/StatefulPartitionedCallConv/StatefulPartitionedCall2@
Conv_1/StatefulPartitionedCallConv_1/StatefulPartitionedCall2@
Conv_2/StatefulPartitionedCallConv_2/StatefulPartitionedCall2R
'fully_connected/StatefulPartitionedCall'fully_connected/StatefulPartitionedCall2V
)fully_connected_1/StatefulPartitionedCall)fully_connected_1/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
!
_user_specified_name	input_1
Ù

!__inference_signature_wrapper_965
input_1,
online_conv_kernel: 
online_conv_bias: .
online_conv_1_kernel: @ 
online_conv_1_bias:@.
online_conv_2_kernel:@@ 
online_conv_2_bias:@1
online_fully_connected_kernel:
À<*
online_fully_connected_bias:	2
online_fully_connected_1_kernel:		+
online_fully_connected_1_bias:	
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinput_1online_conv_kernelonline_conv_biasonline_conv_1_kernelonline_conv_1_biasonline_conv_2_kernelonline_conv_2_biasonline_fully_connected_kernelonline_fully_connected_biasonline_fully_connected_1_kernelonline_fully_connected_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__wrapped_model_7132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿTT: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
!
_user_specified_name	input_1
±

=__inference_Conv_layer_call_and_return_conditional_losses_731

inputsB
(conv2d_readvariableop_online_conv_kernel: 5
'biasadd_readvariableop_online_conv_bias: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOp(conv2d_readvariableop_online_conv_kernel*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp'biasadd_readvariableop_online_conv_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿTT: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
 
_user_specified_nameinputs
B
¦

__inference__wrapped_model_713
input_1N
4online_conv_conv2d_readvariableop_online_conv_kernel: A
3online_conv_biasadd_readvariableop_online_conv_bias: R
8online_conv_1_conv2d_readvariableop_online_conv_1_kernel: @E
7online_conv_1_biasadd_readvariableop_online_conv_1_bias:@R
8online_conv_2_conv2d_readvariableop_online_conv_2_kernel:@@E
7online_conv_2_biasadd_readvariableop_online_conv_2_bias:@^
Jonline_fully_connected_matmul_readvariableop_online_fully_connected_kernel:
À<X
Ionline_fully_connected_biasadd_readvariableop_online_fully_connected_bias:	a
Nonline_fully_connected_1_matmul_readvariableop_online_fully_connected_1_kernel:		[
Monline_fully_connected_1_biasadd_readvariableop_online_fully_connected_1_bias:	
identity¢"Online/Conv/BiasAdd/ReadVariableOp¢!Online/Conv/Conv2D/ReadVariableOp¢$Online/Conv_1/BiasAdd/ReadVariableOp¢#Online/Conv_1/Conv2D/ReadVariableOp¢$Online/Conv_2/BiasAdd/ReadVariableOp¢#Online/Conv_2/Conv2D/ReadVariableOp¢-Online/fully_connected/BiasAdd/ReadVariableOp¢,Online/fully_connected/MatMul/ReadVariableOp¢/Online/fully_connected_1/BiasAdd/ReadVariableOp¢.Online/fully_connected_1/MatMul/ReadVariableOpt
Online/CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT2
Online/Casti
Online/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
Online/truediv/y
Online/truedivRealDivOnline/Cast:y:0Online/truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT2
Online/truedivÃ
!Online/Conv/Conv2D/ReadVariableOpReadVariableOp4online_conv_conv2d_readvariableop_online_conv_kernel*&
_output_shapes
: *
dtype02#
!Online/Conv/Conv2D/ReadVariableOpÓ
Online/Conv/Conv2DConv2DOnline/truediv:z:0)Online/Conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Online/Conv/Conv2D¸
"Online/Conv/BiasAdd/ReadVariableOpReadVariableOp3online_conv_biasadd_readvariableop_online_conv_bias*
_output_shapes
: *
dtype02$
"Online/Conv/BiasAdd/ReadVariableOp¸
Online/Conv/BiasAddBiasAddOnline/Conv/Conv2D:output:0*Online/Conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Online/Conv/BiasAdd
Online/Conv/ReluReluOnline/Conv/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Online/Conv/ReluË
#Online/Conv_1/Conv2D/ReadVariableOpReadVariableOp8online_conv_1_conv2d_readvariableop_online_conv_1_kernel*&
_output_shapes
: @*
dtype02%
#Online/Conv_1/Conv2D/ReadVariableOpå
Online/Conv_1/Conv2DConv2DOnline/Conv/Relu:activations:0+Online/Conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Online/Conv_1/Conv2DÀ
$Online/Conv_1/BiasAdd/ReadVariableOpReadVariableOp7online_conv_1_biasadd_readvariableop_online_conv_1_bias*
_output_shapes
:@*
dtype02&
$Online/Conv_1/BiasAdd/ReadVariableOpÀ
Online/Conv_1/BiasAddBiasAddOnline/Conv_1/Conv2D:output:0,Online/Conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Online/Conv_1/BiasAdd
Online/Conv_1/ReluReluOnline/Conv_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Online/Conv_1/ReluË
#Online/Conv_2/Conv2D/ReadVariableOpReadVariableOp8online_conv_2_conv2d_readvariableop_online_conv_2_kernel*&
_output_shapes
:@@*
dtype02%
#Online/Conv_2/Conv2D/ReadVariableOpç
Online/Conv_2/Conv2DConv2D Online/Conv_1/Relu:activations:0+Online/Conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Online/Conv_2/Conv2DÀ
$Online/Conv_2/BiasAdd/ReadVariableOpReadVariableOp7online_conv_2_biasadd_readvariableop_online_conv_2_bias*
_output_shapes
:@*
dtype02&
$Online/Conv_2/BiasAdd/ReadVariableOpÀ
Online/Conv_2/BiasAddBiasAddOnline/Conv_2/Conv2D:output:0,Online/Conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Online/Conv_2/BiasAdd
Online/Conv_2/ReluReluOnline/Conv_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Online/Conv_2/Relu}
Online/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
Online/flatten/Const¯
Online/flatten/ReshapeReshape Online/Conv_2/Relu:activations:0Online/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<2
Online/flatten/Reshapeé
,Online/fully_connected/MatMul/ReadVariableOpReadVariableOpJonline_fully_connected_matmul_readvariableop_online_fully_connected_kernel* 
_output_shapes
:
À<*
dtype02.
,Online/fully_connected/MatMul/ReadVariableOpÒ
Online/fully_connected/MatMulMatMulOnline/flatten/Reshape:output:04Online/fully_connected/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Online/fully_connected/MatMulå
-Online/fully_connected/BiasAdd/ReadVariableOpReadVariableOpIonline_fully_connected_biasadd_readvariableop_online_fully_connected_bias*
_output_shapes	
:*
dtype02/
-Online/fully_connected/BiasAdd/ReadVariableOpÞ
Online/fully_connected/BiasAddBiasAdd'Online/fully_connected/MatMul:product:05Online/fully_connected/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
Online/fully_connected/BiasAdd
Online/fully_connected/ReluRelu'Online/fully_connected/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Online/fully_connected/Reluð
.Online/fully_connected_1/MatMul/ReadVariableOpReadVariableOpNonline_fully_connected_1_matmul_readvariableop_online_fully_connected_1_kernel*
_output_shapes
:		*
dtype020
.Online/fully_connected_1/MatMul/ReadVariableOpá
Online/fully_connected_1/MatMulMatMul)Online/fully_connected/Relu:activations:06Online/fully_connected_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2!
Online/fully_connected_1/MatMulì
/Online/fully_connected_1/BiasAdd/ReadVariableOpReadVariableOpMonline_fully_connected_1_biasadd_readvariableop_online_fully_connected_1_bias*
_output_shapes
:	*
dtype021
/Online/fully_connected_1/BiasAdd/ReadVariableOpå
 Online/fully_connected_1/BiasAddBiasAdd)Online/fully_connected_1/MatMul:product:07Online/fully_connected_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2"
 Online/fully_connected_1/BiasAdd¢
IdentityIdentity)Online/fully_connected_1/BiasAdd:output:0#^Online/Conv/BiasAdd/ReadVariableOp"^Online/Conv/Conv2D/ReadVariableOp%^Online/Conv_1/BiasAdd/ReadVariableOp$^Online/Conv_1/Conv2D/ReadVariableOp%^Online/Conv_2/BiasAdd/ReadVariableOp$^Online/Conv_2/Conv2D/ReadVariableOp.^Online/fully_connected/BiasAdd/ReadVariableOp-^Online/fully_connected/MatMul/ReadVariableOp0^Online/fully_connected_1/BiasAdd/ReadVariableOp/^Online/fully_connected_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿTT: : : : : : : : : : 2H
"Online/Conv/BiasAdd/ReadVariableOp"Online/Conv/BiasAdd/ReadVariableOp2F
!Online/Conv/Conv2D/ReadVariableOp!Online/Conv/Conv2D/ReadVariableOp2L
$Online/Conv_1/BiasAdd/ReadVariableOp$Online/Conv_1/BiasAdd/ReadVariableOp2J
#Online/Conv_1/Conv2D/ReadVariableOp#Online/Conv_1/Conv2D/ReadVariableOp2L
$Online/Conv_2/BiasAdd/ReadVariableOp$Online/Conv_2/BiasAdd/ReadVariableOp2J
#Online/Conv_2/Conv2D/ReadVariableOp#Online/Conv_2/Conv2D/ReadVariableOp2^
-Online/fully_connected/BiasAdd/ReadVariableOp-Online/fully_connected/BiasAdd/ReadVariableOp2\
,Online/fully_connected/MatMul/ReadVariableOp,Online/fully_connected/MatMul/ReadVariableOp2b
/Online/fully_connected_1/BiasAdd/ReadVariableOp/Online/fully_connected_1/BiasAdd/ReadVariableOp2`
.Online/fully_connected_1/MatMul/ReadVariableOp.Online/fully_connected_1/MatMul/ReadVariableOp:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
!
_user_specified_name	input_1
ý

$__inference_Online_layer_call_fn_842
input_1,
online_conv_kernel: 
online_conv_bias: .
online_conv_1_kernel: @ 
online_conv_1_bias:@.
online_conv_2_kernel:@@ 
online_conv_2_bias:@1
online_fully_connected_kernel:
À<*
online_fully_connected_bias:	2
online_fully_connected_1_kernel:		+
online_fully_connected_1_bias:	
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinput_1online_conv_kernelonline_conv_biasonline_conv_1_kernelonline_conv_1_biasonline_conv_2_kernelonline_conv_2_biasonline_fully_connected_kernelonline_fully_connected_biasonline_fully_connected_1_kernelonline_fully_connected_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_Online_layer_call_and_return_conditional_losses_8292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿTT: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
!
_user_specified_name	input_1
»
®
#__inference_Conv_layer_call_fn_1001

inputs.
online_conv_1_kernel: @ 
online_conv_1_bias:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsonline_conv_1_kernelonline_conv_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_Conv_layer_call_and_return_conditional_losses_7462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ô
Æ
.__inference_fully_connected_layer_call_fn_1048

inputs1
online_fully_connected_kernel:
À<*
online_fully_connected_bias:	
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsonline_fully_connected_kernelonline_fully_connected_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_fully_connected_layer_call_and_return_conditional_losses_7842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ<: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<
 
_user_specified_nameinputs


>__inference_Conv_layer_call_and_return_conditional_losses_1012

inputsD
*conv2d_readvariableop_online_conv_2_kernel:@@7
)biasadd_readvariableop_online_conv_2_bias:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¡
Conv2D/ReadVariableOpReadVariableOp*conv2d_readvariableop_online_conv_2_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp)biasadd_readvariableop_online_conv_2_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¹

=__inference_Conv_layer_call_and_return_conditional_losses_746

inputsD
*conv2d_readvariableop_online_conv_1_kernel: @7
)biasadd_readvariableop_online_conv_1_bias:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¡
Conv2D/ReadVariableOpReadVariableOp*conv2d_readvariableop_online_conv_1_kernel*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp)biasadd_readvariableop_online_conv_1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
°

¦
H__inference_fully_connected_layer_call_and_return_conditional_losses_798

inputsH
5matmul_readvariableop_online_fully_connected_1_kernel:		B
4biasadd_readvariableop_online_fully_connected_1_bias:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¥
MatMul/ReadVariableOpReadVariableOp5matmul_readvariableop_online_fully_connected_1_kernel*
_output_shapes
:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
MatMul¡
BiasAdd/ReadVariableOpReadVariableOp4biasadd_readvariableop_online_fully_connected_1_bias*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
B
&__inference_flatten_layer_call_fn_1030

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
:ÿÿÿÿÿÿÿÿÿÀ<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_7712
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¹

=__inference_Conv_layer_call_and_return_conditional_losses_761

inputsD
*conv2d_readvariableop_online_conv_2_kernel:@@7
)biasadd_readvariableop_online_conv_2_bias:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¡
Conv2D/ReadVariableOpReadVariableOp*conv2d_readvariableop_online_conv_2_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp)biasadd_readvariableop_online_conv_2_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


=__inference_Conv_layer_call_and_return_conditional_losses_976

inputsB
(conv2d_readvariableop_online_conv_kernel: 5
'biasadd_readvariableop_online_conv_bias: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOp(conv2d_readvariableop_online_conv_kernel*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOp'biasadd_readvariableop_online_conv_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿTT: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
 
_user_specified_nameinputs
ã

¥
I__inference_fully_connected_layer_call_and_return_conditional_losses_1041

inputsG
3matmul_readvariableop_online_fully_connected_kernel:
À<A
2biasadd_readvariableop_online_fully_connected_bias:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¤
MatMul/ReadVariableOpReadVariableOp3matmul_readvariableop_online_fully_connected_kernel* 
_output_shapes
:
À<*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul 
BiasAdd/ReadVariableOpReadVariableOp2biasadd_readvariableop_online_fully_connected_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<
 
_user_specified_nameinputs
ª"

__inference__traced_save_1118
file_prefix1
-savev2_online_conv_kernel_read_readvariableop/
+savev2_online_conv_bias_read_readvariableop3
/savev2_online_conv_1_kernel_read_readvariableop1
-savev2_online_conv_1_bias_read_readvariableop3
/savev2_online_conv_2_kernel_read_readvariableop1
-savev2_online_conv_2_bias_read_readvariableop<
8savev2_online_fully_connected_kernel_read_readvariableop:
6savev2_online_fully_connected_bias_read_readvariableop>
:savev2_online_fully_connected_1_kernel_read_readvariableop<
8savev2_online_fully_connected_1_bias_read_readvariableop
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
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
ShardedFilenameµ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ç
value½BºB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slicesÈ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_online_conv_kernel_read_readvariableop+savev2_online_conv_bias_read_readvariableop/savev2_online_conv_1_kernel_read_readvariableop-savev2_online_conv_1_bias_read_readvariableop/savev2_online_conv_2_kernel_read_readvariableop-savev2_online_conv_2_bias_read_readvariableop8savev2_online_fully_connected_kernel_read_readvariableop6savev2_online_fully_connected_bias_read_readvariableop:savev2_online_fully_connected_1_kernel_read_readvariableop8savev2_online_fully_connected_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*
_input_shapesr
p: : : : @:@:@@:@:
À<::		:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
À<:!

_output_shapes	
::%	!

_output_shapes
:		: 


_output_shapes
:	:

_output_shapes
: 
Ø
È
.__inference_fully_connected_layer_call_fn_1065

inputs2
online_fully_connected_1_kernel:		+
online_fully_connected_1_bias:	
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsonline_fully_connected_1_kernelonline_fully_connected_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_fully_connected_layer_call_and_return_conditional_losses_7982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
\
@__inference_flatten_layer_call_and_return_conditional_losses_771

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
²
©
"__inference_Conv_layer_call_fn_983

inputs,
online_conv_kernel: 
online_conv_bias: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsonline_conv_kernelonline_conv_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_Conv_layer_call_and_return_conditional_losses_7312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿTT: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
 
_user_specified_nameinputs
Ù%
²
?__inference_Online_layer_call_and_return_conditional_losses_829
input_11
conv_online_conv_kernel: #
conv_online_conv_bias: 5
conv_1_online_conv_1_kernel: @'
conv_1_online_conv_1_bias:@5
conv_2_online_conv_2_kernel:@@'
conv_2_online_conv_2_bias:@A
-fully_connected_online_fully_connected_kernel:
À<:
+fully_connected_online_fully_connected_bias:	D
1fully_connected_1_online_fully_connected_1_kernel:		=
/fully_connected_1_online_fully_connected_1_bias:	
identity¢Conv/StatefulPartitionedCall¢Conv_1/StatefulPartitionedCall¢Conv_2/StatefulPartitionedCall¢'fully_connected/StatefulPartitionedCall¢)fully_connected_1/StatefulPartitionedCallf
CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT2
Cast[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/yu
truedivRealDivCast:y:0truediv/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT2	
truediv£
Conv/StatefulPartitionedCallStatefulPartitionedCalltruediv:z:0conv_online_conv_kernelconv_online_conv_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_Conv_layer_call_and_return_conditional_losses_7312
Conv/StatefulPartitionedCallÉ
Conv_1/StatefulPartitionedCallStatefulPartitionedCall%Conv/StatefulPartitionedCall:output:0conv_1_online_conv_1_kernelconv_1_online_conv_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_Conv_layer_call_and_return_conditional_losses_7462 
Conv_1/StatefulPartitionedCallË
Conv_2/StatefulPartitionedCallStatefulPartitionedCall'Conv_1/StatefulPartitionedCall:output:0conv_2_online_conv_2_kernelconv_2_online_conv_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_Conv_layer_call_and_return_conditional_losses_7612 
Conv_2/StatefulPartitionedCalló
flatten/PartitionedCallPartitionedCall'Conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_7712
flatten/PartitionedCallþ
'fully_connected/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0-fully_connected_online_fully_connected_kernel+fully_connected_online_fully_connected_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_fully_connected_layer_call_and_return_conditional_losses_7842)
'fully_connected/StatefulPartitionedCall
)fully_connected_1/StatefulPartitionedCallStatefulPartitionedCall0fully_connected/StatefulPartitionedCall:output:01fully_connected_1_online_fully_connected_1_kernel/fully_connected_1_online_fully_connected_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_fully_connected_layer_call_and_return_conditional_losses_7982+
)fully_connected_1/StatefulPartitionedCall½
IdentityIdentity2fully_connected_1/StatefulPartitionedCall:output:0^Conv/StatefulPartitionedCall^Conv_1/StatefulPartitionedCall^Conv_2/StatefulPartitionedCall(^fully_connected/StatefulPartitionedCall*^fully_connected_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿTT: : : : : : : : : : 2<
Conv/StatefulPartitionedCallConv/StatefulPartitionedCall2@
Conv_1/StatefulPartitionedCallConv_1/StatefulPartitionedCall2@
Conv_2/StatefulPartitionedCallConv_2/StatefulPartitionedCall2R
'fully_connected/StatefulPartitionedCall'fully_connected/StatefulPartitionedCall2V
)fully_connected_1/StatefulPartitionedCall)fully_connected_1/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿTT
!
_user_specified_name	input_1
¹
]
A__inference_flatten_layer_call_and_return_conditional_losses_1025

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*³
serving_default
C
input_18
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿTT<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ	tensorflow/serving/predict:
«
	conv1
	conv2
	conv3
flatten

dense1

dense2
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
Q_default_save_signature
*R&call_and_return_all_conditional_losses
S__call__"©
_tf_keras_model{"name": "Online", "trainable": true, "expects_training_arg": false, "dtype": "uint8", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "NatureDQNNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 84, 84, 4]}, "uint8", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "NatureDQNNetwork"}}
ç


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*T&call_and_return_all_conditional_losses
U__call__"Â	
_tf_keras_layer¨	{"name": "Conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "Conv", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 84, 84, 4]}}
é


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*V&call_and_return_all_conditional_losses
W__call__"Ä	
_tf_keras_layerª	{"name": "Conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "Conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 21, 21, 32]}}
ë


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"Æ	
_tf_keras_layer¬	{"name": "Conv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "Conv", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 11, 64]}}

regularization_losses
	variables
 trainable_variables
!	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"
_tf_keras_layeré{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 13}}
	

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
*\&call_and_return_all_conditional_losses
]__call__"ã
_tf_keras_layerÉ{"name": "fully_connected", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "fully_connected", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7744}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 7744]}}
	

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
*^&call_and_return_all_conditional_losses
___call__"á
_tf_keras_layerÇ{"name": "fully_connected", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "fully_connected", "trainable": true, "dtype": "float32", "units": 9, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
 "
trackable_list_wrapper
f
0
1
2
3
4
5
"6
#7
(8
)9"
trackable_list_wrapper
f
0
1
2
3
4
5
"6
#7
(8
)9"
trackable_list_wrapper
Ê
.metrics
/layer_regularization_losses
0non_trainable_variables
1layer_metrics
regularization_losses

2layers
	variables
	trainable_variables
S__call__
Q_default_save_signature
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
,
`serving_default"
signature_map
,:* 2Online/Conv/kernel
: 2Online/Conv/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
3metrics
4layer_regularization_losses
5non_trainable_variables
6layer_metrics
regularization_losses

7layers
	variables
trainable_variables
U__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
.:, @2Online/Conv_1/kernel
 :@2Online/Conv_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
8metrics
9layer_regularization_losses
:non_trainable_variables
;layer_metrics
regularization_losses

<layers
	variables
trainable_variables
W__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
.:,@@2Online/Conv_2/kernel
 :@2Online/Conv_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
=metrics
>layer_regularization_losses
?non_trainable_variables
@layer_metrics
regularization_losses

Alayers
	variables
trainable_variables
Y__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Bmetrics
Clayer_regularization_losses
Dnon_trainable_variables
Elayer_metrics
regularization_losses

Flayers
	variables
 trainable_variables
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
1:/
À<2Online/fully_connected/kernel
*:(2Online/fully_connected/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
­
Gmetrics
Hlayer_regularization_losses
Inon_trainable_variables
Jlayer_metrics
$regularization_losses

Klayers
%	variables
&trainable_variables
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
2:0		2Online/fully_connected_1/kernel
+:)	2Online/fully_connected_1/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
­
Lmetrics
Mlayer_regularization_losses
Nnon_trainable_variables
Olayer_metrics
*regularization_losses

Players
+	variables
,trainable_variables
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
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
ä2á
__inference__wrapped_model_713¾
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
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿTT
2
?__inference_Online_layer_call_and_return_conditional_losses_803Í
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿTT
ù2ö
$__inference_Online_layer_call_fn_842Í
²
FullArgSpec
args
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿTT
ç2ä
=__inference_Conv_layer_call_and_return_conditional_losses_976¢
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
annotationsª *
 
Ì2É
"__inference_Conv_layer_call_fn_983¢
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
annotationsª *
 
ç2ä
=__inference_Conv_layer_call_and_return_conditional_losses_994¢
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
annotationsª *
 
Í2Ê
#__inference_Conv_layer_call_fn_1001¢
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
annotationsª *
 
è2å
>__inference_Conv_layer_call_and_return_conditional_losses_1012¢
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
annotationsª *
 
Í2Ê
#__inference_Conv_layer_call_fn_1019¢
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
annotationsª *
 
ë2è
A__inference_flatten_layer_call_and_return_conditional_losses_1025¢
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
annotationsª *
 
Ð2Í
&__inference_flatten_layer_call_fn_1030¢
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
annotationsª *
 
ó2ð
I__inference_fully_connected_layer_call_and_return_conditional_losses_1041¢
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
annotationsª *
 
Ø2Õ
.__inference_fully_connected_layer_call_fn_1048¢
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
annotationsª *
 
ó2ð
I__inference_fully_connected_layer_call_and_return_conditional_losses_1058¢
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
annotationsª *
 
Ø2Õ
.__inference_fully_connected_layer_call_fn_1065¢
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
annotationsª *
 
ÈBÅ
!__inference_signature_wrapper_965input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ®
>__inference_Conv_layer_call_and_return_conditional_losses_1012l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ­
=__inference_Conv_layer_call_and_return_conditional_losses_976l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿTT
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 ­
=__inference_Conv_layer_call_and_return_conditional_losses_994l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
#__inference_Conv_layer_call_fn_1001_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@
#__inference_Conv_layer_call_fn_1019_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@
"__inference_Conv_layer_call_fn_983_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿTT
ª " ÿÿÿÿÿÿÿÿÿ Ø
?__inference_Online_layer_call_and_return_conditional_losses_803
"#()8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿTT
ª "L¢I
B²?
dqn_network0
q_values$!

0/q_valuesÿÿÿÿÿÿÿÿÿ	
 ±
$__inference_Online_layer_call_fn_842
"#()8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿTT
ª "@²=
dqn_network.
q_values"
q_valuesÿÿÿÿÿÿÿÿÿ	
__inference__wrapped_model_713{
"#()8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿTT
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ	¦
A__inference_flatten_layer_call_and_return_conditional_losses_1025a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ<
 ~
&__inference_flatten_layer_call_fn_1030T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿÀ<«
I__inference_fully_connected_layer_call_and_return_conditional_losses_1041^"#0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ<
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ª
I__inference_fully_connected_layer_call_and_return_conditional_losses_1058]()0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ	
 
.__inference_fully_connected_layer_call_fn_1048Q"#0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ<
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_fully_connected_layer_call_fn_1065P()0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ	¬
!__inference_signature_wrapper_965
"#()C¢@
¢ 
9ª6
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿTT"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ	