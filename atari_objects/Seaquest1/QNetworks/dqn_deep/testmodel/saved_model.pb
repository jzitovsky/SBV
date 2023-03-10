��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
�
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
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring �
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718֧
�
Online/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameOnline/conv2d/kernel
�
(Online/conv2d/kernel/Read/ReadVariableOpReadVariableOpOnline/conv2d/kernel*&
_output_shapes
:0*
dtype0
|
Online/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*#
shared_nameOnline/conv2d/bias
u
&Online/conv2d/bias/Read/ReadVariableOpReadVariableOpOnline/conv2d/bias*
_output_shapes
:0*
dtype0
�
Online/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:00*'
shared_nameOnline/conv2d_1/kernel
�
*Online/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpOnline/conv2d_1/kernel*&
_output_shapes
:00*
dtype0
�
Online/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameOnline/conv2d_1/bias
y
(Online/conv2d_1/bias/Read/ReadVariableOpReadVariableOpOnline/conv2d_1/bias*
_output_shapes
:0*
dtype0
�
!Online/conv_stack/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0H*2
shared_name#!Online/conv_stack/conv2d_2/kernel
�
5Online/conv_stack/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp!Online/conv_stack/conv2d_2/kernel*&
_output_shapes
:0H*
dtype0
�
Online/conv_stack/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*0
shared_name!Online/conv_stack/conv2d_2/bias
�
3Online/conv_stack/conv2d_2/bias/Read/ReadVariableOpReadVariableOpOnline/conv_stack/conv2d_2/bias*
_output_shapes
:H*
dtype0
�
!Online/conv_stack/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:HH*2
shared_name#!Online/conv_stack/conv2d_3/kernel
�
5Online/conv_stack/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp!Online/conv_stack/conv2d_3/kernel*&
_output_shapes
:HH*
dtype0
�
Online/conv_stack/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*0
shared_name!Online/conv_stack/conv2d_3/bias
�
3Online/conv_stack/conv2d_3/bias/Read/ReadVariableOpReadVariableOpOnline/conv_stack/conv2d_3/bias*
_output_shapes
:H*
dtype0
�
!Online/conv_stack/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0H*2
shared_name#!Online/conv_stack/conv2d_4/kernel
�
5Online/conv_stack/conv2d_4/kernel/Read/ReadVariableOpReadVariableOp!Online/conv_stack/conv2d_4/kernel*&
_output_shapes
:0H*
dtype0
�
Online/conv_stack/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*0
shared_name!Online/conv_stack/conv2d_4/bias
�
3Online/conv_stack/conv2d_4/bias/Read/ReadVariableOpReadVariableOpOnline/conv_stack/conv2d_4/bias*
_output_shapes
:H*
dtype0
�
#Online/conv_stack_1/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:HH*4
shared_name%#Online/conv_stack_1/conv2d_5/kernel
�
7Online/conv_stack_1/conv2d_5/kernel/Read/ReadVariableOpReadVariableOp#Online/conv_stack_1/conv2d_5/kernel*&
_output_shapes
:HH*
dtype0
�
!Online/conv_stack_1/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*2
shared_name#!Online/conv_stack_1/conv2d_5/bias
�
5Online/conv_stack_1/conv2d_5/bias/Read/ReadVariableOpReadVariableOp!Online/conv_stack_1/conv2d_5/bias*
_output_shapes
:H*
dtype0
�
#Online/conv_stack_1/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:HH*4
shared_name%#Online/conv_stack_1/conv2d_6/kernel
�
7Online/conv_stack_1/conv2d_6/kernel/Read/ReadVariableOpReadVariableOp#Online/conv_stack_1/conv2d_6/kernel*&
_output_shapes
:HH*
dtype0
�
!Online/conv_stack_1/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*2
shared_name#!Online/conv_stack_1/conv2d_6/bias
�
5Online/conv_stack_1/conv2d_6/bias/Read/ReadVariableOpReadVariableOp!Online/conv_stack_1/conv2d_6/bias*
_output_shapes
:H*
dtype0
�
#Online/conv_stack_2/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:H�*4
shared_name%#Online/conv_stack_2/conv2d_7/kernel
�
7Online/conv_stack_2/conv2d_7/kernel/Read/ReadVariableOpReadVariableOp#Online/conv_stack_2/conv2d_7/kernel*'
_output_shapes
:H�*
dtype0
�
!Online/conv_stack_2/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Online/conv_stack_2/conv2d_7/bias
�
5Online/conv_stack_2/conv2d_7/bias/Read/ReadVariableOpReadVariableOp!Online/conv_stack_2/conv2d_7/bias*
_output_shapes	
:�*
dtype0
�
#Online/conv_stack_2/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*4
shared_name%#Online/conv_stack_2/conv2d_8/kernel
�
7Online/conv_stack_2/conv2d_8/kernel/Read/ReadVariableOpReadVariableOp#Online/conv_stack_2/conv2d_8/kernel*(
_output_shapes
:��*
dtype0
�
!Online/conv_stack_2/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Online/conv_stack_2/conv2d_8/bias
�
5Online/conv_stack_2/conv2d_8/bias/Read/ReadVariableOpReadVariableOp!Online/conv_stack_2/conv2d_8/bias*
_output_shapes	
:�*
dtype0
�
#Online/conv_stack_2/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:H�*4
shared_name%#Online/conv_stack_2/conv2d_9/kernel
�
7Online/conv_stack_2/conv2d_9/kernel/Read/ReadVariableOpReadVariableOp#Online/conv_stack_2/conv2d_9/kernel*'
_output_shapes
:H�*
dtype0
�
!Online/conv_stack_2/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Online/conv_stack_2/conv2d_9/bias
�
5Online/conv_stack_2/conv2d_9/bias/Read/ReadVariableOpReadVariableOp!Online/conv_stack_2/conv2d_9/bias*
_output_shapes	
:�*
dtype0
�
$Online/conv_stack_3/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*5
shared_name&$Online/conv_stack_3/conv2d_10/kernel
�
8Online/conv_stack_3/conv2d_10/kernel/Read/ReadVariableOpReadVariableOp$Online/conv_stack_3/conv2d_10/kernel*(
_output_shapes
:��*
dtype0
�
"Online/conv_stack_3/conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Online/conv_stack_3/conv2d_10/bias
�
6Online/conv_stack_3/conv2d_10/bias/Read/ReadVariableOpReadVariableOp"Online/conv_stack_3/conv2d_10/bias*
_output_shapes	
:�*
dtype0
�
$Online/conv_stack_3/conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*5
shared_name&$Online/conv_stack_3/conv2d_11/kernel
�
8Online/conv_stack_3/conv2d_11/kernel/Read/ReadVariableOpReadVariableOp$Online/conv_stack_3/conv2d_11/kernel*(
_output_shapes
:��*
dtype0
�
"Online/conv_stack_3/conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Online/conv_stack_3/conv2d_11/bias
�
6Online/conv_stack_3/conv2d_11/bias/Read/ReadVariableOpReadVariableOp"Online/conv_stack_3/conv2d_11/bias*
_output_shapes	
:�*
dtype0
�
$Online/conv_stack_4/conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*5
shared_name&$Online/conv_stack_4/conv2d_12/kernel
�
8Online/conv_stack_4/conv2d_12/kernel/Read/ReadVariableOpReadVariableOp$Online/conv_stack_4/conv2d_12/kernel*(
_output_shapes
:��*
dtype0
�
"Online/conv_stack_4/conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Online/conv_stack_4/conv2d_12/bias
�
6Online/conv_stack_4/conv2d_12/bias/Read/ReadVariableOpReadVariableOp"Online/conv_stack_4/conv2d_12/bias*
_output_shapes	
:�*
dtype0
�
$Online/conv_stack_4/conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*5
shared_name&$Online/conv_stack_4/conv2d_13/kernel
�
8Online/conv_stack_4/conv2d_13/kernel/Read/ReadVariableOpReadVariableOp$Online/conv_stack_4/conv2d_13/kernel*(
_output_shapes
:��*
dtype0
�
"Online/conv_stack_4/conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Online/conv_stack_4/conv2d_13/bias
�
6Online/conv_stack_4/conv2d_13/bias/Read/ReadVariableOpReadVariableOp"Online/conv_stack_4/conv2d_13/bias*
_output_shapes	
:�*
dtype0
�
$Online/conv_stack_4/conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*5
shared_name&$Online/conv_stack_4/conv2d_14/kernel
�
8Online/conv_stack_4/conv2d_14/kernel/Read/ReadVariableOpReadVariableOp$Online/conv_stack_4/conv2d_14/kernel*(
_output_shapes
:��*
dtype0
�
"Online/conv_stack_4/conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Online/conv_stack_4/conv2d_14/bias
�
6Online/conv_stack_4/conv2d_14/bias/Read/ReadVariableOpReadVariableOp"Online/conv_stack_4/conv2d_14/bias*
_output_shapes	
:�*
dtype0
�
$Online/conv_stack_5/conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*5
shared_name&$Online/conv_stack_5/conv2d_15/kernel
�
8Online/conv_stack_5/conv2d_15/kernel/Read/ReadVariableOpReadVariableOp$Online/conv_stack_5/conv2d_15/kernel*(
_output_shapes
:��*
dtype0
�
"Online/conv_stack_5/conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Online/conv_stack_5/conv2d_15/bias
�
6Online/conv_stack_5/conv2d_15/bias/Read/ReadVariableOpReadVariableOp"Online/conv_stack_5/conv2d_15/bias*
_output_shapes	
:�*
dtype0
�
$Online/conv_stack_5/conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*5
shared_name&$Online/conv_stack_5/conv2d_16/kernel
�
8Online/conv_stack_5/conv2d_16/kernel/Read/ReadVariableOpReadVariableOp$Online/conv_stack_5/conv2d_16/kernel*(
_output_shapes
:��*
dtype0
�
"Online/conv_stack_5/conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Online/conv_stack_5/conv2d_16/bias
�
6Online/conv_stack_5/conv2d_16/bias/Read/ReadVariableOpReadVariableOp"Online/conv_stack_5/conv2d_16/bias*
_output_shapes	
:�*
dtype0
�
Online/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameOnline/dense/kernel
|
'Online/dense/kernel/Read/ReadVariableOpReadVariableOpOnline/dense/kernel*
_output_shapes
:	�*
dtype0
z
Online/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameOnline/dense/bias
s
%Online/dense/bias/Read/ReadVariableOpReadVariableOpOnline/dense/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Ң
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
s
main_layers
regularization_losses
	variables
trainable_variables
	keras_api

signatures
^
0
1
	2

3
4
5
6
7
8
9
10
11
12
 
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
)21
*22
+23
,24
-25
.26
/27
028
129
230
331
432
533
634
735
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
)21
*22
+23
,24
-25
.26
/27
028
129
230
331
432
533
634
735
�
regularization_losses

8layers
	variables
9layer_metrics
:metrics
;non_trainable_variables
<layer_regularization_losses
trainable_variables
 
h

kernel
bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
R
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
h

kernel
bias
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
R
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
R
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
�
Qmain_layers
Rskip_layers
Sjoin_layers
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
�
Xmain_layers
Yskip_layers
Zjoin_layers
[regularization_losses
\	variables
]trainable_variables
^	keras_api
�
_main_layers
`skip_layers
ajoin_layers
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
�
fmain_layers
gskip_layers
hjoin_layers
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
�
mmain_layers
nskip_layers
ojoin_layers
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
�
tmain_layers
uskip_layers
vjoin_layers
wregularization_losses
x	variables
ytrainable_variables
z	keras_api
R
{regularization_losses
|	variables
}trainable_variables
~	keras_api
k

6kernel
7bias
regularization_losses
�	variables
�trainable_variables
�	keras_api
PN
VARIABLE_VALUEOnline/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEOnline/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEOnline/conv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEOnline/conv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!Online/conv_stack/conv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEOnline/conv_stack/conv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!Online/conv_stack/conv2d_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEOnline/conv_stack/conv2d_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!Online/conv_stack/conv2d_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEOnline/conv_stack/conv2d_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#Online/conv_stack_1/conv2d_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!Online/conv_stack_1/conv2d_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#Online/conv_stack_1/conv2d_6/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!Online/conv_stack_1/conv2d_6/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#Online/conv_stack_2/conv2d_7/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!Online/conv_stack_2/conv2d_7/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#Online/conv_stack_2/conv2d_8/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!Online/conv_stack_2/conv2d_8/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#Online/conv_stack_2/conv2d_9/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!Online/conv_stack_2/conv2d_9/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$Online/conv_stack_3/conv2d_10/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"Online/conv_stack_3/conv2d_10/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$Online/conv_stack_3/conv2d_11/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"Online/conv_stack_3/conv2d_11/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$Online/conv_stack_4/conv2d_12/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"Online/conv_stack_4/conv2d_12/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$Online/conv_stack_4/conv2d_13/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"Online/conv_stack_4/conv2d_13/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$Online/conv_stack_4/conv2d_14/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"Online/conv_stack_4/conv2d_14/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$Online/conv_stack_5/conv2d_15/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"Online/conv_stack_5/conv2d_15/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$Online/conv_stack_5/conv2d_16/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"Online/conv_stack_5/conv2d_16/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEOnline/dense/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEOnline/dense/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE
^
0
1
	2

3
4
5
6
7
8
9
10
11
12
 
 
 
 
 

0
1

0
1
�
=regularization_losses
�layers
>	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
?trainable_variables
 
 
 
�
Aregularization_losses
�layers
B	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
Ctrainable_variables
 

0
1

0
1
�
Eregularization_losses
�layers
F	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
Gtrainable_variables
 
 
 
�
Iregularization_losses
�layers
J	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
Ktrainable_variables
 
 
 
�
Mregularization_losses
�layers
N	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
Otrainable_variables
 
�0
�1
�2
�3

�0

�0
�1
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
�
Tregularization_losses
�layers
U	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
Vtrainable_variables

�0
�1
�2
 

�0
�1
 

0
1
 2
!3

0
1
 2
!3
�
[regularization_losses
�layers
\	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
]trainable_variables
 
�0
�1
�2
�3

�0

�0
�1
 
*
"0
#1
$2
%3
&4
'5
*
"0
#1
$2
%3
&4
'5
�
bregularization_losses
�layers
c	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
dtrainable_variables

�0
�1
�2
 

�0
�1
 

(0
)1
*2
+3

(0
)1
*2
+3
�
iregularization_losses
�layers
j	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
ktrainable_variables
 
�0
�1
�2
�3

�0

�0
�1
 
*
,0
-1
.2
/3
04
15
*
,0
-1
.2
/3
04
15
�
pregularization_losses
�layers
q	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
rtrainable_variables

�0
�1
�2
 

�0
�1
 

20
31
42
53

20
31
42
53
�
wregularization_losses
�layers
x	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
ytrainable_variables
 
 
 
�
{regularization_losses
�layers
|	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
}trainable_variables
 

60
71

60
71
�
regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
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
l

kernel
bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
l

kernel
bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
l

kernel
bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
8
�0
�1
�2
�3
�4
�5
�6
 
 
 
 
l

kernel
bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
l

 kernel
!bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
(
�0
�1
�2
�3
�4
 
 
 
 
l

"kernel
#bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
l

$kernel
%bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
l

&kernel
'bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
8
�0
�1
�2
�3
�4
�5
�6
 
 
 
 
l

(kernel
)bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
l

*kernel
+bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
(
�0
�1
�2
�3
�4
 
 
 
 
l

,kernel
-bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
l

.kernel
/bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
l

0kernel
1bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
8
�0
�1
�2
�3
�4
�5
�6
 
 
 
 
l

2kernel
3bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
l

4kernel
5bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
(
�0
�1
�2
�3
�4
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

0
1

0
1
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 

0
1

0
1
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 

0
1

0
1
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 

0
1

0
1
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 

 0
!1

 0
!1
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 

"0
#1

"0
#1
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 

$0
%1

$0
%1
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 

&0
'1

&0
'1
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 

(0
)1

(0
)1
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 

*0
+1

*0
+1
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 

,0
-1

,0
-1
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 

.0
/1

.0
/1
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 

00
11

00
11
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 

20
31

20
31
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 

40
51

40
51
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
 
 
 
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
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
�
serving_default_input_1Placeholder*/
_output_shapes
:���������TT*
dtype0*$
shape:���������TT
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Online/conv2d/kernelOnline/conv2d/biasOnline/conv2d_1/kernelOnline/conv2d_1/bias!Online/conv_stack/conv2d_2/kernelOnline/conv_stack/conv2d_2/bias!Online/conv_stack/conv2d_3/kernelOnline/conv_stack/conv2d_3/bias!Online/conv_stack/conv2d_4/kernelOnline/conv_stack/conv2d_4/bias#Online/conv_stack_1/conv2d_5/kernel!Online/conv_stack_1/conv2d_5/bias#Online/conv_stack_1/conv2d_6/kernel!Online/conv_stack_1/conv2d_6/bias#Online/conv_stack_2/conv2d_7/kernel!Online/conv_stack_2/conv2d_7/bias#Online/conv_stack_2/conv2d_8/kernel!Online/conv_stack_2/conv2d_8/bias#Online/conv_stack_2/conv2d_9/kernel!Online/conv_stack_2/conv2d_9/bias$Online/conv_stack_3/conv2d_10/kernel"Online/conv_stack_3/conv2d_10/bias$Online/conv_stack_3/conv2d_11/kernel"Online/conv_stack_3/conv2d_11/bias$Online/conv_stack_4/conv2d_12/kernel"Online/conv_stack_4/conv2d_12/bias$Online/conv_stack_4/conv2d_13/kernel"Online/conv_stack_4/conv2d_13/bias$Online/conv_stack_4/conv2d_14/kernel"Online/conv_stack_4/conv2d_14/bias$Online/conv_stack_5/conv2d_15/kernel"Online/conv_stack_5/conv2d_15/bias$Online/conv_stack_5/conv2d_16/kernel"Online/conv_stack_5/conv2d_16/biasOnline/dense/kernelOnline/dense/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference_signature_wrapper_3075
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(Online/conv2d/kernel/Read/ReadVariableOp&Online/conv2d/bias/Read/ReadVariableOp*Online/conv2d_1/kernel/Read/ReadVariableOp(Online/conv2d_1/bias/Read/ReadVariableOp5Online/conv_stack/conv2d_2/kernel/Read/ReadVariableOp3Online/conv_stack/conv2d_2/bias/Read/ReadVariableOp5Online/conv_stack/conv2d_3/kernel/Read/ReadVariableOp3Online/conv_stack/conv2d_3/bias/Read/ReadVariableOp5Online/conv_stack/conv2d_4/kernel/Read/ReadVariableOp3Online/conv_stack/conv2d_4/bias/Read/ReadVariableOp7Online/conv_stack_1/conv2d_5/kernel/Read/ReadVariableOp5Online/conv_stack_1/conv2d_5/bias/Read/ReadVariableOp7Online/conv_stack_1/conv2d_6/kernel/Read/ReadVariableOp5Online/conv_stack_1/conv2d_6/bias/Read/ReadVariableOp7Online/conv_stack_2/conv2d_7/kernel/Read/ReadVariableOp5Online/conv_stack_2/conv2d_7/bias/Read/ReadVariableOp7Online/conv_stack_2/conv2d_8/kernel/Read/ReadVariableOp5Online/conv_stack_2/conv2d_8/bias/Read/ReadVariableOp7Online/conv_stack_2/conv2d_9/kernel/Read/ReadVariableOp5Online/conv_stack_2/conv2d_9/bias/Read/ReadVariableOp8Online/conv_stack_3/conv2d_10/kernel/Read/ReadVariableOp6Online/conv_stack_3/conv2d_10/bias/Read/ReadVariableOp8Online/conv_stack_3/conv2d_11/kernel/Read/ReadVariableOp6Online/conv_stack_3/conv2d_11/bias/Read/ReadVariableOp8Online/conv_stack_4/conv2d_12/kernel/Read/ReadVariableOp6Online/conv_stack_4/conv2d_12/bias/Read/ReadVariableOp8Online/conv_stack_4/conv2d_13/kernel/Read/ReadVariableOp6Online/conv_stack_4/conv2d_13/bias/Read/ReadVariableOp8Online/conv_stack_4/conv2d_14/kernel/Read/ReadVariableOp6Online/conv_stack_4/conv2d_14/bias/Read/ReadVariableOp8Online/conv_stack_5/conv2d_15/kernel/Read/ReadVariableOp6Online/conv_stack_5/conv2d_15/bias/Read/ReadVariableOp8Online/conv_stack_5/conv2d_16/kernel/Read/ReadVariableOp6Online/conv_stack_5/conv2d_16/bias/Read/ReadVariableOp'Online/dense/kernel/Read/ReadVariableOp%Online/dense/bias/Read/ReadVariableOpConst*1
Tin*
(2&*
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
GPU2*0J 8� *&
f!R
__inference__traced_save_3472
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameOnline/conv2d/kernelOnline/conv2d/biasOnline/conv2d_1/kernelOnline/conv2d_1/bias!Online/conv_stack/conv2d_2/kernelOnline/conv_stack/conv2d_2/bias!Online/conv_stack/conv2d_3/kernelOnline/conv_stack/conv2d_3/bias!Online/conv_stack/conv2d_4/kernelOnline/conv_stack/conv2d_4/bias#Online/conv_stack_1/conv2d_5/kernel!Online/conv_stack_1/conv2d_5/bias#Online/conv_stack_1/conv2d_6/kernel!Online/conv_stack_1/conv2d_6/bias#Online/conv_stack_2/conv2d_7/kernel!Online/conv_stack_2/conv2d_7/bias#Online/conv_stack_2/conv2d_8/kernel!Online/conv_stack_2/conv2d_8/bias#Online/conv_stack_2/conv2d_9/kernel!Online/conv_stack_2/conv2d_9/bias$Online/conv_stack_3/conv2d_10/kernel"Online/conv_stack_3/conv2d_10/bias$Online/conv_stack_3/conv2d_11/kernel"Online/conv_stack_3/conv2d_11/bias$Online/conv_stack_4/conv2d_12/kernel"Online/conv_stack_4/conv2d_12/bias$Online/conv_stack_4/conv2d_13/kernel"Online/conv_stack_4/conv2d_13/bias$Online/conv_stack_4/conv2d_14/kernel"Online/conv_stack_4/conv2d_14/bias$Online/conv_stack_5/conv2d_15/kernel"Online/conv_stack_5/conv2d_15/bias$Online/conv_stack_5/conv2d_16/kernel"Online/conv_stack_5/conv2d_16/biasOnline/dense/kernelOnline/dense/bias*0
Tin)
'2%*
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
GPU2*0J 8� *)
f$R"
 __inference__traced_restore_3590Ԁ
� 
�
%__inference_Online_layer_call_fn_2755
input_1.
online_conv2d_kernel:0 
online_conv2d_bias:00
online_conv2d_1_kernel:00"
online_conv2d_1_bias:0;
!online_conv_stack_conv2d_2_kernel:0H-
online_conv_stack_conv2d_2_bias:H;
!online_conv_stack_conv2d_3_kernel:HH-
online_conv_stack_conv2d_3_bias:H;
!online_conv_stack_conv2d_4_kernel:0H-
online_conv_stack_conv2d_4_bias:H=
#online_conv_stack_1_conv2d_5_kernel:HH/
!online_conv_stack_1_conv2d_5_bias:H=
#online_conv_stack_1_conv2d_6_kernel:HH/
!online_conv_stack_1_conv2d_6_bias:H>
#online_conv_stack_2_conv2d_7_kernel:H�0
!online_conv_stack_2_conv2d_7_bias:	�?
#online_conv_stack_2_conv2d_8_kernel:��0
!online_conv_stack_2_conv2d_8_bias:	�>
#online_conv_stack_2_conv2d_9_kernel:H�0
!online_conv_stack_2_conv2d_9_bias:	�@
$online_conv_stack_3_conv2d_10_kernel:��1
"online_conv_stack_3_conv2d_10_bias:	�@
$online_conv_stack_3_conv2d_11_kernel:��1
"online_conv_stack_3_conv2d_11_bias:	�@
$online_conv_stack_4_conv2d_12_kernel:��1
"online_conv_stack_4_conv2d_12_bias:	�@
$online_conv_stack_4_conv2d_13_kernel:��1
"online_conv_stack_4_conv2d_13_bias:	�@
$online_conv_stack_4_conv2d_14_kernel:��1
"online_conv_stack_4_conv2d_14_bias:	�@
$online_conv_stack_5_conv2d_15_kernel:��1
"online_conv_stack_5_conv2d_15_bias:	�@
$online_conv_stack_5_conv2d_16_kernel:��1
"online_conv_stack_5_conv2d_16_bias:	�&
online_dense_kernel:	�
online_dense_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1online_conv2d_kernelonline_conv2d_biasonline_conv2d_1_kernelonline_conv2d_1_bias!online_conv_stack_conv2d_2_kernelonline_conv_stack_conv2d_2_bias!online_conv_stack_conv2d_3_kernelonline_conv_stack_conv2d_3_bias!online_conv_stack_conv2d_4_kernelonline_conv_stack_conv2d_4_bias#online_conv_stack_1_conv2d_5_kernel!online_conv_stack_1_conv2d_5_bias#online_conv_stack_1_conv2d_6_kernel!online_conv_stack_1_conv2d_6_bias#online_conv_stack_2_conv2d_7_kernel!online_conv_stack_2_conv2d_7_bias#online_conv_stack_2_conv2d_8_kernel!online_conv_stack_2_conv2d_8_bias#online_conv_stack_2_conv2d_9_kernel!online_conv_stack_2_conv2d_9_bias$online_conv_stack_3_conv2d_10_kernel"online_conv_stack_3_conv2d_10_bias$online_conv_stack_3_conv2d_11_kernel"online_conv_stack_3_conv2d_11_bias$online_conv_stack_4_conv2d_12_kernel"online_conv_stack_4_conv2d_12_bias$online_conv_stack_4_conv2d_13_kernel"online_conv_stack_4_conv2d_13_bias$online_conv_stack_4_conv2d_14_kernel"online_conv_stack_4_conv2d_14_bias$online_conv_stack_5_conv2d_15_kernel"online_conv_stack_5_conv2d_15_bias$online_conv_stack_5_conv2d_16_kernel"online_conv_stack_5_conv2d_16_biasonline_dense_kernelonline_dense_bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_Online_layer_call_and_return_conditional_losses_27162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*v
_input_shapese
c:���������TT: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������TT
!
_user_specified_name	input_1
�
J
.__inference_max_pooling2d_3_layer_call_fn_2395

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_23922
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
+__inference_conv_stack_1_layer_call_fn_3194

inputs=
#online_conv_stack_1_conv2d_5_kernel:HH/
!online_conv_stack_1_conv2d_5_bias:H=
#online_conv_stack_1_conv2d_6_kernel:HH/
!online_conv_stack_1_conv2d_6_bias:H
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs#online_conv_stack_1_conv2d_5_kernel!online_conv_stack_1_conv2d_5_bias#online_conv_stack_1_conv2d_6_kernel!online_conv_stack_1_conv2d_6_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_1_layer_call_and_return_conditional_losses_25172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������H2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������H: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
n
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_2402

inputs
identity�
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_3112

inputsF
,conv2d_readvariableop_online_conv2d_1_kernel:009
+biasadd_readvariableop_online_conv2d_1_bias:0
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOp,conv2d_readvariableop_online_conv2d_1_kernel*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT0*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOp+biasadd_readvariableop_online_conv2d_1_bias*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT02	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������TT02

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������TT0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������TT0
 
_user_specified_nameinputs
�%
�
D__inference_conv_stack_layer_call_and_return_conditional_losses_2490

inputsZ
@conv2d_2_conv2d_readvariableop_online_conv_stack_conv2d_2_kernel:0HM
?conv2d_2_biasadd_readvariableop_online_conv_stack_conv2d_2_bias:HZ
@conv2d_3_conv2d_readvariableop_online_conv_stack_conv2d_3_kernel:HHM
?conv2d_3_biasadd_readvariableop_online_conv_stack_conv2d_3_bias:HZ
@conv2d_4_conv2d_readvariableop_online_conv_stack_conv2d_4_kernel:0HM
?conv2d_4_biasadd_readvariableop_online_conv_stack_conv2d_4_bias:H
identity��conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp@conv2d_2_conv2d_readvariableop_online_conv_stack_conv2d_2_kernel*&
_output_shapes
:0H*
dtype02 
conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������))H*
paddingSAME*
strides
2
conv2d_2/Conv2D�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp?conv2d_2_biasadd_readvariableop_online_conv_stack_conv2d_2_bias*
_output_shapes
:H*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������))H2
conv2d_2/BiasAdd�
leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_2/BiasAdd:output:0*/
_output_shapes
:���������))H*
alpha%���>2
leaky_re_lu_2/LeakyRelu�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp@conv2d_3_conv2d_readvariableop_online_conv_stack_conv2d_3_kernel*&
_output_shapes
:HH*
dtype02 
conv2d_3/Conv2D/ReadVariableOp�
conv2d_3/Conv2DConv2D%leaky_re_lu_2/LeakyRelu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������))H*
paddingSAME*
strides
2
conv2d_3/Conv2D�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp?conv2d_3_biasadd_readvariableop_online_conv_stack_conv2d_3_bias*
_output_shapes
:H*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������))H2
conv2d_3/BiasAdd�
max_pooling2d_1/MaxPoolMaxPoolconv2d_3/BiasAdd:output:0*/
_output_shapes
:���������H*
ksize
*
paddingSAME*
strides
2
max_pooling2d_1/MaxPool�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp@conv2d_4_conv2d_readvariableop_online_conv_stack_conv2d_4_kernel*&
_output_shapes
:0H*
dtype02 
conv2d_4/Conv2D/ReadVariableOp�
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
2
conv2d_4/Conv2D�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp?conv2d_4_biasadd_readvariableop_online_conv_stack_conv2d_4_bias*
_output_shapes
:H*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H2
conv2d_4/BiasAdd�
add/addAddV2 max_pooling2d_1/MaxPool:output:0conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������H2	
add/add�
leaky_re_lu_3/LeakyRelu	LeakyReluadd/add:z:0*/
_output_shapes
:���������H*
alpha%���>2
leaky_re_lu_3/LeakyRelu�
IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������))0: : : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������))0
 
_user_specified_nameinputs
�
a
E__inference_leaky_re_lu_layer_call_and_return_conditional_losses_3097

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������TT0*
alpha%���>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������TT02

Identity"
identityIdentity:output:0*.
_input_shapes
:���������TT0:W S
/
_output_shapes
:���������TT0
 
_user_specified_nameinputs
�$
�
D__inference_conv_stack_layer_call_and_return_conditional_losses_3155

inputsZ
@conv2d_2_conv2d_readvariableop_online_conv_stack_conv2d_2_kernel:0HM
?conv2d_2_biasadd_readvariableop_online_conv_stack_conv2d_2_bias:HZ
@conv2d_3_conv2d_readvariableop_online_conv_stack_conv2d_3_kernel:HHM
?conv2d_3_biasadd_readvariableop_online_conv_stack_conv2d_3_bias:HZ
@conv2d_4_conv2d_readvariableop_online_conv_stack_conv2d_4_kernel:0HM
?conv2d_4_biasadd_readvariableop_online_conv_stack_conv2d_4_bias:H
identity��conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp@conv2d_2_conv2d_readvariableop_online_conv_stack_conv2d_2_kernel*&
_output_shapes
:0H*
dtype02 
conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������))H*
paddingSAME*
strides
2
conv2d_2/Conv2D�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp?conv2d_2_biasadd_readvariableop_online_conv_stack_conv2d_2_bias*
_output_shapes
:H*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������))H2
conv2d_2/BiasAdd�
leaky_re_lu_2/LeakyRelu	LeakyReluconv2d_2/BiasAdd:output:0*/
_output_shapes
:���������))H*
alpha%���>2
leaky_re_lu_2/LeakyRelu�
conv2d_3/Conv2D/ReadVariableOpReadVariableOp@conv2d_3_conv2d_readvariableop_online_conv_stack_conv2d_3_kernel*&
_output_shapes
:HH*
dtype02 
conv2d_3/Conv2D/ReadVariableOp�
conv2d_3/Conv2DConv2D%leaky_re_lu_2/LeakyRelu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������))H*
paddingSAME*
strides
2
conv2d_3/Conv2D�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp?conv2d_3_biasadd_readvariableop_online_conv_stack_conv2d_3_bias*
_output_shapes
:H*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������))H2
conv2d_3/BiasAdd�
max_pooling2d_1/MaxPoolMaxPoolconv2d_3/BiasAdd:output:0*/
_output_shapes
:���������H*
ksize
*
paddingSAME*
strides
2
max_pooling2d_1/MaxPool�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp@conv2d_4_conv2d_readvariableop_online_conv_stack_conv2d_4_kernel*&
_output_shapes
:0H*
dtype02 
conv2d_4/Conv2D/ReadVariableOp�
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
2
conv2d_4/Conv2D�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp?conv2d_4_biasadd_readvariableop_online_conv_stack_conv2d_4_bias*
_output_shapes
:H*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H2
conv2d_4/BiasAdd�
add/addAddV2 max_pooling2d_1/MaxPool:output:0conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������H2	
add/add�
leaky_re_lu_3/LeakyRelu	LeakyReluadd/add:z:0*/
_output_shapes
:���������H*
alpha%���>2
leaky_re_lu_3/LeakyRelu�
IdentityIdentity%leaky_re_lu_3/LeakyRelu:activations:0 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������H2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������))0: : : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������))0
 
_user_specified_nameinputs
�
�+
__inference__wrapped_model_2327
input_1R
8online_conv2d_conv2d_readvariableop_online_conv2d_kernel:0E
7online_conv2d_biasadd_readvariableop_online_conv2d_bias:0V
<online_conv2d_1_conv2d_readvariableop_online_conv2d_1_kernel:00I
;online_conv2d_1_biasadd_readvariableop_online_conv2d_1_bias:0l
Ronline_conv_stack_conv2d_2_conv2d_readvariableop_online_conv_stack_conv2d_2_kernel:0H_
Qonline_conv_stack_conv2d_2_biasadd_readvariableop_online_conv_stack_conv2d_2_bias:Hl
Ronline_conv_stack_conv2d_3_conv2d_readvariableop_online_conv_stack_conv2d_3_kernel:HH_
Qonline_conv_stack_conv2d_3_biasadd_readvariableop_online_conv_stack_conv2d_3_bias:Hl
Ronline_conv_stack_conv2d_4_conv2d_readvariableop_online_conv_stack_conv2d_4_kernel:0H_
Qonline_conv_stack_conv2d_4_biasadd_readvariableop_online_conv_stack_conv2d_4_bias:Hp
Vonline_conv_stack_1_conv2d_5_conv2d_readvariableop_online_conv_stack_1_conv2d_5_kernel:HHc
Uonline_conv_stack_1_conv2d_5_biasadd_readvariableop_online_conv_stack_1_conv2d_5_bias:Hp
Vonline_conv_stack_1_conv2d_6_conv2d_readvariableop_online_conv_stack_1_conv2d_6_kernel:HHc
Uonline_conv_stack_1_conv2d_6_biasadd_readvariableop_online_conv_stack_1_conv2d_6_bias:Hq
Vonline_conv_stack_2_conv2d_7_conv2d_readvariableop_online_conv_stack_2_conv2d_7_kernel:H�d
Uonline_conv_stack_2_conv2d_7_biasadd_readvariableop_online_conv_stack_2_conv2d_7_bias:	�r
Vonline_conv_stack_2_conv2d_8_conv2d_readvariableop_online_conv_stack_2_conv2d_8_kernel:��d
Uonline_conv_stack_2_conv2d_8_biasadd_readvariableop_online_conv_stack_2_conv2d_8_bias:	�q
Vonline_conv_stack_2_conv2d_9_conv2d_readvariableop_online_conv_stack_2_conv2d_9_kernel:H�d
Uonline_conv_stack_2_conv2d_9_biasadd_readvariableop_online_conv_stack_2_conv2d_9_bias:	�t
Xonline_conv_stack_3_conv2d_10_conv2d_readvariableop_online_conv_stack_3_conv2d_10_kernel:��f
Wonline_conv_stack_3_conv2d_10_biasadd_readvariableop_online_conv_stack_3_conv2d_10_bias:	�t
Xonline_conv_stack_3_conv2d_11_conv2d_readvariableop_online_conv_stack_3_conv2d_11_kernel:��f
Wonline_conv_stack_3_conv2d_11_biasadd_readvariableop_online_conv_stack_3_conv2d_11_bias:	�t
Xonline_conv_stack_4_conv2d_12_conv2d_readvariableop_online_conv_stack_4_conv2d_12_kernel:��f
Wonline_conv_stack_4_conv2d_12_biasadd_readvariableop_online_conv_stack_4_conv2d_12_bias:	�t
Xonline_conv_stack_4_conv2d_13_conv2d_readvariableop_online_conv_stack_4_conv2d_13_kernel:��f
Wonline_conv_stack_4_conv2d_13_biasadd_readvariableop_online_conv_stack_4_conv2d_13_bias:	�t
Xonline_conv_stack_4_conv2d_14_conv2d_readvariableop_online_conv_stack_4_conv2d_14_kernel:��f
Wonline_conv_stack_4_conv2d_14_biasadd_readvariableop_online_conv_stack_4_conv2d_14_bias:	�t
Xonline_conv_stack_5_conv2d_15_conv2d_readvariableop_online_conv_stack_5_conv2d_15_kernel:��f
Wonline_conv_stack_5_conv2d_15_biasadd_readvariableop_online_conv_stack_5_conv2d_15_bias:	�t
Xonline_conv_stack_5_conv2d_16_conv2d_readvariableop_online_conv_stack_5_conv2d_16_kernel:��f
Wonline_conv_stack_5_conv2d_16_biasadd_readvariableop_online_conv_stack_5_conv2d_16_bias:	�I
6online_dense_matmul_readvariableop_online_dense_kernel:	�C
5online_dense_biasadd_readvariableop_online_dense_bias:
identity��$Online/conv2d/BiasAdd/ReadVariableOp�#Online/conv2d/Conv2D/ReadVariableOp�&Online/conv2d_1/BiasAdd/ReadVariableOp�%Online/conv2d_1/Conv2D/ReadVariableOp�1Online/conv_stack/conv2d_2/BiasAdd/ReadVariableOp�0Online/conv_stack/conv2d_2/Conv2D/ReadVariableOp�1Online/conv_stack/conv2d_3/BiasAdd/ReadVariableOp�0Online/conv_stack/conv2d_3/Conv2D/ReadVariableOp�1Online/conv_stack/conv2d_4/BiasAdd/ReadVariableOp�0Online/conv_stack/conv2d_4/Conv2D/ReadVariableOp�3Online/conv_stack_1/conv2d_5/BiasAdd/ReadVariableOp�2Online/conv_stack_1/conv2d_5/Conv2D/ReadVariableOp�3Online/conv_stack_1/conv2d_6/BiasAdd/ReadVariableOp�2Online/conv_stack_1/conv2d_6/Conv2D/ReadVariableOp�3Online/conv_stack_2/conv2d_7/BiasAdd/ReadVariableOp�2Online/conv_stack_2/conv2d_7/Conv2D/ReadVariableOp�3Online/conv_stack_2/conv2d_8/BiasAdd/ReadVariableOp�2Online/conv_stack_2/conv2d_8/Conv2D/ReadVariableOp�3Online/conv_stack_2/conv2d_9/BiasAdd/ReadVariableOp�2Online/conv_stack_2/conv2d_9/Conv2D/ReadVariableOp�4Online/conv_stack_3/conv2d_10/BiasAdd/ReadVariableOp�3Online/conv_stack_3/conv2d_10/Conv2D/ReadVariableOp�4Online/conv_stack_3/conv2d_11/BiasAdd/ReadVariableOp�3Online/conv_stack_3/conv2d_11/Conv2D/ReadVariableOp�4Online/conv_stack_4/conv2d_12/BiasAdd/ReadVariableOp�3Online/conv_stack_4/conv2d_12/Conv2D/ReadVariableOp�4Online/conv_stack_4/conv2d_13/BiasAdd/ReadVariableOp�3Online/conv_stack_4/conv2d_13/Conv2D/ReadVariableOp�4Online/conv_stack_4/conv2d_14/BiasAdd/ReadVariableOp�3Online/conv_stack_4/conv2d_14/Conv2D/ReadVariableOp�4Online/conv_stack_5/conv2d_15/BiasAdd/ReadVariableOp�3Online/conv_stack_5/conv2d_15/Conv2D/ReadVariableOp�4Online/conv_stack_5/conv2d_16/BiasAdd/ReadVariableOp�3Online/conv_stack_5/conv2d_16/Conv2D/ReadVariableOp�#Online/dense/BiasAdd/ReadVariableOp�"Online/dense/MatMul/ReadVariableOpt
Online/CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:���������TT2
Online/Casti
Online/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
Online/truediv/y�
Online/truedivRealDivOnline/Cast:y:0Online/truediv/y:output:0*
T0*/
_output_shapes
:���������TT2
Online/truediv�
#Online/conv2d/Conv2D/ReadVariableOpReadVariableOp8online_conv2d_conv2d_readvariableop_online_conv2d_kernel*&
_output_shapes
:0*
dtype02%
#Online/conv2d/Conv2D/ReadVariableOp�
Online/conv2d/Conv2DConv2DOnline/truediv:z:0+Online/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT0*
paddingSAME*
strides
2
Online/conv2d/Conv2D�
$Online/conv2d/BiasAdd/ReadVariableOpReadVariableOp7online_conv2d_biasadd_readvariableop_online_conv2d_bias*
_output_shapes
:0*
dtype02&
$Online/conv2d/BiasAdd/ReadVariableOp�
Online/conv2d/BiasAddBiasAddOnline/conv2d/Conv2D:output:0,Online/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT02
Online/conv2d/BiasAdd�
Online/leaky_re_lu/LeakyRelu	LeakyReluOnline/conv2d/BiasAdd:output:0*/
_output_shapes
:���������TT0*
alpha%���>2
Online/leaky_re_lu/LeakyRelu�
%Online/conv2d_1/Conv2D/ReadVariableOpReadVariableOp<online_conv2d_1_conv2d_readvariableop_online_conv2d_1_kernel*&
_output_shapes
:00*
dtype02'
%Online/conv2d_1/Conv2D/ReadVariableOp�
Online/conv2d_1/Conv2DConv2D*Online/leaky_re_lu/LeakyRelu:activations:0-Online/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT0*
paddingSAME*
strides
2
Online/conv2d_1/Conv2D�
&Online/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp;online_conv2d_1_biasadd_readvariableop_online_conv2d_1_bias*
_output_shapes
:0*
dtype02(
&Online/conv2d_1/BiasAdd/ReadVariableOp�
Online/conv2d_1/BiasAddBiasAddOnline/conv2d_1/Conv2D:output:0.Online/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT02
Online/conv2d_1/BiasAdd�
Online/max_pooling2d/MaxPoolMaxPool Online/conv2d_1/BiasAdd:output:0*/
_output_shapes
:���������))0*
ksize
*
paddingVALID*
strides
2
Online/max_pooling2d/MaxPool�
Online/leaky_re_lu_1/LeakyRelu	LeakyRelu%Online/max_pooling2d/MaxPool:output:0*/
_output_shapes
:���������))0*
alpha%���>2 
Online/leaky_re_lu_1/LeakyRelu�
0Online/conv_stack/conv2d_2/Conv2D/ReadVariableOpReadVariableOpRonline_conv_stack_conv2d_2_conv2d_readvariableop_online_conv_stack_conv2d_2_kernel*&
_output_shapes
:0H*
dtype022
0Online/conv_stack/conv2d_2/Conv2D/ReadVariableOp�
!Online/conv_stack/conv2d_2/Conv2DConv2D,Online/leaky_re_lu_1/LeakyRelu:activations:08Online/conv_stack/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������))H*
paddingSAME*
strides
2#
!Online/conv_stack/conv2d_2/Conv2D�
1Online/conv_stack/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpQonline_conv_stack_conv2d_2_biasadd_readvariableop_online_conv_stack_conv2d_2_bias*
_output_shapes
:H*
dtype023
1Online/conv_stack/conv2d_2/BiasAdd/ReadVariableOp�
"Online/conv_stack/conv2d_2/BiasAddBiasAdd*Online/conv_stack/conv2d_2/Conv2D:output:09Online/conv_stack/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������))H2$
"Online/conv_stack/conv2d_2/BiasAdd�
)Online/conv_stack/leaky_re_lu_2/LeakyRelu	LeakyRelu+Online/conv_stack/conv2d_2/BiasAdd:output:0*/
_output_shapes
:���������))H*
alpha%���>2+
)Online/conv_stack/leaky_re_lu_2/LeakyRelu�
0Online/conv_stack/conv2d_3/Conv2D/ReadVariableOpReadVariableOpRonline_conv_stack_conv2d_3_conv2d_readvariableop_online_conv_stack_conv2d_3_kernel*&
_output_shapes
:HH*
dtype022
0Online/conv_stack/conv2d_3/Conv2D/ReadVariableOp�
!Online/conv_stack/conv2d_3/Conv2DConv2D7Online/conv_stack/leaky_re_lu_2/LeakyRelu:activations:08Online/conv_stack/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������))H*
paddingSAME*
strides
2#
!Online/conv_stack/conv2d_3/Conv2D�
1Online/conv_stack/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpQonline_conv_stack_conv2d_3_biasadd_readvariableop_online_conv_stack_conv2d_3_bias*
_output_shapes
:H*
dtype023
1Online/conv_stack/conv2d_3/BiasAdd/ReadVariableOp�
"Online/conv_stack/conv2d_3/BiasAddBiasAdd*Online/conv_stack/conv2d_3/Conv2D:output:09Online/conv_stack/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������))H2$
"Online/conv_stack/conv2d_3/BiasAdd�
)Online/conv_stack/max_pooling2d_1/MaxPoolMaxPool+Online/conv_stack/conv2d_3/BiasAdd:output:0*/
_output_shapes
:���������H*
ksize
*
paddingSAME*
strides
2+
)Online/conv_stack/max_pooling2d_1/MaxPool�
0Online/conv_stack/conv2d_4/Conv2D/ReadVariableOpReadVariableOpRonline_conv_stack_conv2d_4_conv2d_readvariableop_online_conv_stack_conv2d_4_kernel*&
_output_shapes
:0H*
dtype022
0Online/conv_stack/conv2d_4/Conv2D/ReadVariableOp�
!Online/conv_stack/conv2d_4/Conv2DConv2D,Online/leaky_re_lu_1/LeakyRelu:activations:08Online/conv_stack/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
2#
!Online/conv_stack/conv2d_4/Conv2D�
1Online/conv_stack/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpQonline_conv_stack_conv2d_4_biasadd_readvariableop_online_conv_stack_conv2d_4_bias*
_output_shapes
:H*
dtype023
1Online/conv_stack/conv2d_4/BiasAdd/ReadVariableOp�
"Online/conv_stack/conv2d_4/BiasAddBiasAdd*Online/conv_stack/conv2d_4/Conv2D:output:09Online/conv_stack/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H2$
"Online/conv_stack/conv2d_4/BiasAdd�
Online/conv_stack/add/addAddV22Online/conv_stack/max_pooling2d_1/MaxPool:output:0+Online/conv_stack/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:���������H2
Online/conv_stack/add/add�
)Online/conv_stack/leaky_re_lu_3/LeakyRelu	LeakyReluOnline/conv_stack/add/add:z:0*/
_output_shapes
:���������H*
alpha%���>2+
)Online/conv_stack/leaky_re_lu_3/LeakyRelu�
2Online/conv_stack_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOpVonline_conv_stack_1_conv2d_5_conv2d_readvariableop_online_conv_stack_1_conv2d_5_kernel*&
_output_shapes
:HH*
dtype024
2Online/conv_stack_1/conv2d_5/Conv2D/ReadVariableOp�
#Online/conv_stack_1/conv2d_5/Conv2DConv2D7Online/conv_stack/leaky_re_lu_3/LeakyRelu:activations:0:Online/conv_stack_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
2%
#Online/conv_stack_1/conv2d_5/Conv2D�
3Online/conv_stack_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpUonline_conv_stack_1_conv2d_5_biasadd_readvariableop_online_conv_stack_1_conv2d_5_bias*
_output_shapes
:H*
dtype025
3Online/conv_stack_1/conv2d_5/BiasAdd/ReadVariableOp�
$Online/conv_stack_1/conv2d_5/BiasAddBiasAdd,Online/conv_stack_1/conv2d_5/Conv2D:output:0;Online/conv_stack_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H2&
$Online/conv_stack_1/conv2d_5/BiasAdd�
+Online/conv_stack_1/leaky_re_lu_4/LeakyRelu	LeakyRelu-Online/conv_stack_1/conv2d_5/BiasAdd:output:0*/
_output_shapes
:���������H*
alpha%���>2-
+Online/conv_stack_1/leaky_re_lu_4/LeakyRelu�
2Online/conv_stack_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOpVonline_conv_stack_1_conv2d_6_conv2d_readvariableop_online_conv_stack_1_conv2d_6_kernel*&
_output_shapes
:HH*
dtype024
2Online/conv_stack_1/conv2d_6/Conv2D/ReadVariableOp�
#Online/conv_stack_1/conv2d_6/Conv2DConv2D9Online/conv_stack_1/leaky_re_lu_4/LeakyRelu:activations:0:Online/conv_stack_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
2%
#Online/conv_stack_1/conv2d_6/Conv2D�
3Online/conv_stack_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpUonline_conv_stack_1_conv2d_6_biasadd_readvariableop_online_conv_stack_1_conv2d_6_bias*
_output_shapes
:H*
dtype025
3Online/conv_stack_1/conv2d_6/BiasAdd/ReadVariableOp�
$Online/conv_stack_1/conv2d_6/BiasAddBiasAdd,Online/conv_stack_1/conv2d_6/Conv2D:output:0;Online/conv_stack_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H2&
$Online/conv_stack_1/conv2d_6/BiasAdd�
Online/conv_stack_1/add_1/addAddV2-Online/conv_stack_1/conv2d_6/BiasAdd:output:07Online/conv_stack/leaky_re_lu_3/LeakyRelu:activations:0*
T0*/
_output_shapes
:���������H2
Online/conv_stack_1/add_1/add�
+Online/conv_stack_1/leaky_re_lu_5/LeakyRelu	LeakyRelu!Online/conv_stack_1/add_1/add:z:0*/
_output_shapes
:���������H*
alpha%���>2-
+Online/conv_stack_1/leaky_re_lu_5/LeakyRelu�
2Online/conv_stack_2/conv2d_7/Conv2D/ReadVariableOpReadVariableOpVonline_conv_stack_2_conv2d_7_conv2d_readvariableop_online_conv_stack_2_conv2d_7_kernel*'
_output_shapes
:H�*
dtype024
2Online/conv_stack_2/conv2d_7/Conv2D/ReadVariableOp�
#Online/conv_stack_2/conv2d_7/Conv2DConv2D9Online/conv_stack_1/leaky_re_lu_5/LeakyRelu:activations:0:Online/conv_stack_2/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2%
#Online/conv_stack_2/conv2d_7/Conv2D�
3Online/conv_stack_2/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpUonline_conv_stack_2_conv2d_7_biasadd_readvariableop_online_conv_stack_2_conv2d_7_bias*
_output_shapes	
:�*
dtype025
3Online/conv_stack_2/conv2d_7/BiasAdd/ReadVariableOp�
$Online/conv_stack_2/conv2d_7/BiasAddBiasAdd,Online/conv_stack_2/conv2d_7/Conv2D:output:0;Online/conv_stack_2/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2&
$Online/conv_stack_2/conv2d_7/BiasAdd�
+Online/conv_stack_2/leaky_re_lu_6/LeakyRelu	LeakyRelu-Online/conv_stack_2/conv2d_7/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2-
+Online/conv_stack_2/leaky_re_lu_6/LeakyRelu�
2Online/conv_stack_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOpVonline_conv_stack_2_conv2d_8_conv2d_readvariableop_online_conv_stack_2_conv2d_8_kernel*(
_output_shapes
:��*
dtype024
2Online/conv_stack_2/conv2d_8/Conv2D/ReadVariableOp�
#Online/conv_stack_2/conv2d_8/Conv2DConv2D9Online/conv_stack_2/leaky_re_lu_6/LeakyRelu:activations:0:Online/conv_stack_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2%
#Online/conv_stack_2/conv2d_8/Conv2D�
3Online/conv_stack_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpUonline_conv_stack_2_conv2d_8_biasadd_readvariableop_online_conv_stack_2_conv2d_8_bias*
_output_shapes	
:�*
dtype025
3Online/conv_stack_2/conv2d_8/BiasAdd/ReadVariableOp�
$Online/conv_stack_2/conv2d_8/BiasAddBiasAdd,Online/conv_stack_2/conv2d_8/Conv2D:output:0;Online/conv_stack_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2&
$Online/conv_stack_2/conv2d_8/BiasAdd�
+Online/conv_stack_2/max_pooling2d_2/MaxPoolMaxPool-Online/conv_stack_2/conv2d_8/BiasAdd:output:0*0
_output_shapes
:����������*
ksize
*
paddingSAME*
strides
2-
+Online/conv_stack_2/max_pooling2d_2/MaxPool�
2Online/conv_stack_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOpVonline_conv_stack_2_conv2d_9_conv2d_readvariableop_online_conv_stack_2_conv2d_9_kernel*'
_output_shapes
:H�*
dtype024
2Online/conv_stack_2/conv2d_9/Conv2D/ReadVariableOp�
#Online/conv_stack_2/conv2d_9/Conv2DConv2D9Online/conv_stack_1/leaky_re_lu_5/LeakyRelu:activations:0:Online/conv_stack_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2%
#Online/conv_stack_2/conv2d_9/Conv2D�
3Online/conv_stack_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOpUonline_conv_stack_2_conv2d_9_biasadd_readvariableop_online_conv_stack_2_conv2d_9_bias*
_output_shapes	
:�*
dtype025
3Online/conv_stack_2/conv2d_9/BiasAdd/ReadVariableOp�
$Online/conv_stack_2/conv2d_9/BiasAddBiasAdd,Online/conv_stack_2/conv2d_9/Conv2D:output:0;Online/conv_stack_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2&
$Online/conv_stack_2/conv2d_9/BiasAdd�
Online/conv_stack_2/add_2/addAddV24Online/conv_stack_2/max_pooling2d_2/MaxPool:output:0-Online/conv_stack_2/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
Online/conv_stack_2/add_2/add�
+Online/conv_stack_2/leaky_re_lu_7/LeakyRelu	LeakyRelu!Online/conv_stack_2/add_2/add:z:0*0
_output_shapes
:����������*
alpha%���>2-
+Online/conv_stack_2/leaky_re_lu_7/LeakyRelu�
3Online/conv_stack_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOpXonline_conv_stack_3_conv2d_10_conv2d_readvariableop_online_conv_stack_3_conv2d_10_kernel*(
_output_shapes
:��*
dtype025
3Online/conv_stack_3/conv2d_10/Conv2D/ReadVariableOp�
$Online/conv_stack_3/conv2d_10/Conv2DConv2D9Online/conv_stack_2/leaky_re_lu_7/LeakyRelu:activations:0;Online/conv_stack_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2&
$Online/conv_stack_3/conv2d_10/Conv2D�
4Online/conv_stack_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOpWonline_conv_stack_3_conv2d_10_biasadd_readvariableop_online_conv_stack_3_conv2d_10_bias*
_output_shapes	
:�*
dtype026
4Online/conv_stack_3/conv2d_10/BiasAdd/ReadVariableOp�
%Online/conv_stack_3/conv2d_10/BiasAddBiasAdd-Online/conv_stack_3/conv2d_10/Conv2D:output:0<Online/conv_stack_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2'
%Online/conv_stack_3/conv2d_10/BiasAdd�
+Online/conv_stack_3/leaky_re_lu_8/LeakyRelu	LeakyRelu.Online/conv_stack_3/conv2d_10/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2-
+Online/conv_stack_3/leaky_re_lu_8/LeakyRelu�
3Online/conv_stack_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOpXonline_conv_stack_3_conv2d_11_conv2d_readvariableop_online_conv_stack_3_conv2d_11_kernel*(
_output_shapes
:��*
dtype025
3Online/conv_stack_3/conv2d_11/Conv2D/ReadVariableOp�
$Online/conv_stack_3/conv2d_11/Conv2DConv2D9Online/conv_stack_3/leaky_re_lu_8/LeakyRelu:activations:0;Online/conv_stack_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2&
$Online/conv_stack_3/conv2d_11/Conv2D�
4Online/conv_stack_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOpWonline_conv_stack_3_conv2d_11_biasadd_readvariableop_online_conv_stack_3_conv2d_11_bias*
_output_shapes	
:�*
dtype026
4Online/conv_stack_3/conv2d_11/BiasAdd/ReadVariableOp�
%Online/conv_stack_3/conv2d_11/BiasAddBiasAdd-Online/conv_stack_3/conv2d_11/Conv2D:output:0<Online/conv_stack_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2'
%Online/conv_stack_3/conv2d_11/BiasAdd�
Online/conv_stack_3/add_3/addAddV2.Online/conv_stack_3/conv2d_11/BiasAdd:output:09Online/conv_stack_2/leaky_re_lu_7/LeakyRelu:activations:0*
T0*0
_output_shapes
:����������2
Online/conv_stack_3/add_3/add�
+Online/conv_stack_3/leaky_re_lu_9/LeakyRelu	LeakyRelu!Online/conv_stack_3/add_3/add:z:0*0
_output_shapes
:����������*
alpha%���>2-
+Online/conv_stack_3/leaky_re_lu_9/LeakyRelu�
3Online/conv_stack_4/conv2d_12/Conv2D/ReadVariableOpReadVariableOpXonline_conv_stack_4_conv2d_12_conv2d_readvariableop_online_conv_stack_4_conv2d_12_kernel*(
_output_shapes
:��*
dtype025
3Online/conv_stack_4/conv2d_12/Conv2D/ReadVariableOp�
$Online/conv_stack_4/conv2d_12/Conv2DConv2D9Online/conv_stack_3/leaky_re_lu_9/LeakyRelu:activations:0;Online/conv_stack_4/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2&
$Online/conv_stack_4/conv2d_12/Conv2D�
4Online/conv_stack_4/conv2d_12/BiasAdd/ReadVariableOpReadVariableOpWonline_conv_stack_4_conv2d_12_biasadd_readvariableop_online_conv_stack_4_conv2d_12_bias*
_output_shapes	
:�*
dtype026
4Online/conv_stack_4/conv2d_12/BiasAdd/ReadVariableOp�
%Online/conv_stack_4/conv2d_12/BiasAddBiasAdd-Online/conv_stack_4/conv2d_12/Conv2D:output:0<Online/conv_stack_4/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2'
%Online/conv_stack_4/conv2d_12/BiasAdd�
,Online/conv_stack_4/leaky_re_lu_10/LeakyRelu	LeakyRelu.Online/conv_stack_4/conv2d_12/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2.
,Online/conv_stack_4/leaky_re_lu_10/LeakyRelu�
3Online/conv_stack_4/conv2d_13/Conv2D/ReadVariableOpReadVariableOpXonline_conv_stack_4_conv2d_13_conv2d_readvariableop_online_conv_stack_4_conv2d_13_kernel*(
_output_shapes
:��*
dtype025
3Online/conv_stack_4/conv2d_13/Conv2D/ReadVariableOp�
$Online/conv_stack_4/conv2d_13/Conv2DConv2D:Online/conv_stack_4/leaky_re_lu_10/LeakyRelu:activations:0;Online/conv_stack_4/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2&
$Online/conv_stack_4/conv2d_13/Conv2D�
4Online/conv_stack_4/conv2d_13/BiasAdd/ReadVariableOpReadVariableOpWonline_conv_stack_4_conv2d_13_biasadd_readvariableop_online_conv_stack_4_conv2d_13_bias*
_output_shapes	
:�*
dtype026
4Online/conv_stack_4/conv2d_13/BiasAdd/ReadVariableOp�
%Online/conv_stack_4/conv2d_13/BiasAddBiasAdd-Online/conv_stack_4/conv2d_13/Conv2D:output:0<Online/conv_stack_4/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2'
%Online/conv_stack_4/conv2d_13/BiasAdd�
+Online/conv_stack_4/max_pooling2d_3/MaxPoolMaxPool.Online/conv_stack_4/conv2d_13/BiasAdd:output:0*0
_output_shapes
:����������*
ksize
*
paddingSAME*
strides
2-
+Online/conv_stack_4/max_pooling2d_3/MaxPool�
3Online/conv_stack_4/conv2d_14/Conv2D/ReadVariableOpReadVariableOpXonline_conv_stack_4_conv2d_14_conv2d_readvariableop_online_conv_stack_4_conv2d_14_kernel*(
_output_shapes
:��*
dtype025
3Online/conv_stack_4/conv2d_14/Conv2D/ReadVariableOp�
$Online/conv_stack_4/conv2d_14/Conv2DConv2D9Online/conv_stack_3/leaky_re_lu_9/LeakyRelu:activations:0;Online/conv_stack_4/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2&
$Online/conv_stack_4/conv2d_14/Conv2D�
4Online/conv_stack_4/conv2d_14/BiasAdd/ReadVariableOpReadVariableOpWonline_conv_stack_4_conv2d_14_biasadd_readvariableop_online_conv_stack_4_conv2d_14_bias*
_output_shapes	
:�*
dtype026
4Online/conv_stack_4/conv2d_14/BiasAdd/ReadVariableOp�
%Online/conv_stack_4/conv2d_14/BiasAddBiasAdd-Online/conv_stack_4/conv2d_14/Conv2D:output:0<Online/conv_stack_4/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2'
%Online/conv_stack_4/conv2d_14/BiasAdd�
Online/conv_stack_4/add_4/addAddV24Online/conv_stack_4/max_pooling2d_3/MaxPool:output:0.Online/conv_stack_4/conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
Online/conv_stack_4/add_4/add�
,Online/conv_stack_4/leaky_re_lu_11/LeakyRelu	LeakyRelu!Online/conv_stack_4/add_4/add:z:0*0
_output_shapes
:����������*
alpha%���>2.
,Online/conv_stack_4/leaky_re_lu_11/LeakyRelu�
3Online/conv_stack_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOpXonline_conv_stack_5_conv2d_15_conv2d_readvariableop_online_conv_stack_5_conv2d_15_kernel*(
_output_shapes
:��*
dtype025
3Online/conv_stack_5/conv2d_15/Conv2D/ReadVariableOp�
$Online/conv_stack_5/conv2d_15/Conv2DConv2D:Online/conv_stack_4/leaky_re_lu_11/LeakyRelu:activations:0;Online/conv_stack_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2&
$Online/conv_stack_5/conv2d_15/Conv2D�
4Online/conv_stack_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOpWonline_conv_stack_5_conv2d_15_biasadd_readvariableop_online_conv_stack_5_conv2d_15_bias*
_output_shapes	
:�*
dtype026
4Online/conv_stack_5/conv2d_15/BiasAdd/ReadVariableOp�
%Online/conv_stack_5/conv2d_15/BiasAddBiasAdd-Online/conv_stack_5/conv2d_15/Conv2D:output:0<Online/conv_stack_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2'
%Online/conv_stack_5/conv2d_15/BiasAdd�
,Online/conv_stack_5/leaky_re_lu_12/LeakyRelu	LeakyRelu.Online/conv_stack_5/conv2d_15/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2.
,Online/conv_stack_5/leaky_re_lu_12/LeakyRelu�
3Online/conv_stack_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOpXonline_conv_stack_5_conv2d_16_conv2d_readvariableop_online_conv_stack_5_conv2d_16_kernel*(
_output_shapes
:��*
dtype025
3Online/conv_stack_5/conv2d_16/Conv2D/ReadVariableOp�
$Online/conv_stack_5/conv2d_16/Conv2DConv2D:Online/conv_stack_5/leaky_re_lu_12/LeakyRelu:activations:0;Online/conv_stack_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2&
$Online/conv_stack_5/conv2d_16/Conv2D�
4Online/conv_stack_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOpWonline_conv_stack_5_conv2d_16_biasadd_readvariableop_online_conv_stack_5_conv2d_16_bias*
_output_shapes	
:�*
dtype026
4Online/conv_stack_5/conv2d_16/BiasAdd/ReadVariableOp�
%Online/conv_stack_5/conv2d_16/BiasAddBiasAdd-Online/conv_stack_5/conv2d_16/Conv2D:output:0<Online/conv_stack_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2'
%Online/conv_stack_5/conv2d_16/BiasAdd�
Online/conv_stack_5/add_5/addAddV2.Online/conv_stack_5/conv2d_16/BiasAdd:output:0:Online/conv_stack_4/leaky_re_lu_11/LeakyRelu:activations:0*
T0*0
_output_shapes
:����������2
Online/conv_stack_5/add_5/add�
,Online/conv_stack_5/leaky_re_lu_13/LeakyRelu	LeakyRelu!Online/conv_stack_5/add_5/add:z:0*0
_output_shapes
:����������*
alpha%���>2.
,Online/conv_stack_5/leaky_re_lu_13/LeakyRelu�
6Online/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      28
6Online/global_average_pooling2d/Mean/reduction_indices�
$Online/global_average_pooling2d/MeanMean:Online/conv_stack_5/leaky_re_lu_13/LeakyRelu:activations:0?Online/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2&
$Online/global_average_pooling2d/Mean�
"Online/dense/MatMul/ReadVariableOpReadVariableOp6online_dense_matmul_readvariableop_online_dense_kernel*
_output_shapes
:	�*
dtype02$
"Online/dense/MatMul/ReadVariableOp�
Online/dense/MatMulMatMul-Online/global_average_pooling2d/Mean:output:0*Online/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Online/dense/MatMul�
#Online/dense/BiasAdd/ReadVariableOpReadVariableOp5online_dense_biasadd_readvariableop_online_dense_bias*
_output_shapes
:*
dtype02%
#Online/dense/BiasAdd/ReadVariableOp�
Online/dense/BiasAddBiasAddOnline/dense/MatMul:product:0+Online/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Online/dense/BiasAdd�
IdentityIdentityOnline/dense/BiasAdd:output:0%^Online/conv2d/BiasAdd/ReadVariableOp$^Online/conv2d/Conv2D/ReadVariableOp'^Online/conv2d_1/BiasAdd/ReadVariableOp&^Online/conv2d_1/Conv2D/ReadVariableOp2^Online/conv_stack/conv2d_2/BiasAdd/ReadVariableOp1^Online/conv_stack/conv2d_2/Conv2D/ReadVariableOp2^Online/conv_stack/conv2d_3/BiasAdd/ReadVariableOp1^Online/conv_stack/conv2d_3/Conv2D/ReadVariableOp2^Online/conv_stack/conv2d_4/BiasAdd/ReadVariableOp1^Online/conv_stack/conv2d_4/Conv2D/ReadVariableOp4^Online/conv_stack_1/conv2d_5/BiasAdd/ReadVariableOp3^Online/conv_stack_1/conv2d_5/Conv2D/ReadVariableOp4^Online/conv_stack_1/conv2d_6/BiasAdd/ReadVariableOp3^Online/conv_stack_1/conv2d_6/Conv2D/ReadVariableOp4^Online/conv_stack_2/conv2d_7/BiasAdd/ReadVariableOp3^Online/conv_stack_2/conv2d_7/Conv2D/ReadVariableOp4^Online/conv_stack_2/conv2d_8/BiasAdd/ReadVariableOp3^Online/conv_stack_2/conv2d_8/Conv2D/ReadVariableOp4^Online/conv_stack_2/conv2d_9/BiasAdd/ReadVariableOp3^Online/conv_stack_2/conv2d_9/Conv2D/ReadVariableOp5^Online/conv_stack_3/conv2d_10/BiasAdd/ReadVariableOp4^Online/conv_stack_3/conv2d_10/Conv2D/ReadVariableOp5^Online/conv_stack_3/conv2d_11/BiasAdd/ReadVariableOp4^Online/conv_stack_3/conv2d_11/Conv2D/ReadVariableOp5^Online/conv_stack_4/conv2d_12/BiasAdd/ReadVariableOp4^Online/conv_stack_4/conv2d_12/Conv2D/ReadVariableOp5^Online/conv_stack_4/conv2d_13/BiasAdd/ReadVariableOp4^Online/conv_stack_4/conv2d_13/Conv2D/ReadVariableOp5^Online/conv_stack_4/conv2d_14/BiasAdd/ReadVariableOp4^Online/conv_stack_4/conv2d_14/Conv2D/ReadVariableOp5^Online/conv_stack_5/conv2d_15/BiasAdd/ReadVariableOp4^Online/conv_stack_5/conv2d_15/Conv2D/ReadVariableOp5^Online/conv_stack_5/conv2d_16/BiasAdd/ReadVariableOp4^Online/conv_stack_5/conv2d_16/Conv2D/ReadVariableOp$^Online/dense/BiasAdd/ReadVariableOp#^Online/dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*v
_input_shapese
c:���������TT: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$Online/conv2d/BiasAdd/ReadVariableOp$Online/conv2d/BiasAdd/ReadVariableOp2J
#Online/conv2d/Conv2D/ReadVariableOp#Online/conv2d/Conv2D/ReadVariableOp2P
&Online/conv2d_1/BiasAdd/ReadVariableOp&Online/conv2d_1/BiasAdd/ReadVariableOp2N
%Online/conv2d_1/Conv2D/ReadVariableOp%Online/conv2d_1/Conv2D/ReadVariableOp2f
1Online/conv_stack/conv2d_2/BiasAdd/ReadVariableOp1Online/conv_stack/conv2d_2/BiasAdd/ReadVariableOp2d
0Online/conv_stack/conv2d_2/Conv2D/ReadVariableOp0Online/conv_stack/conv2d_2/Conv2D/ReadVariableOp2f
1Online/conv_stack/conv2d_3/BiasAdd/ReadVariableOp1Online/conv_stack/conv2d_3/BiasAdd/ReadVariableOp2d
0Online/conv_stack/conv2d_3/Conv2D/ReadVariableOp0Online/conv_stack/conv2d_3/Conv2D/ReadVariableOp2f
1Online/conv_stack/conv2d_4/BiasAdd/ReadVariableOp1Online/conv_stack/conv2d_4/BiasAdd/ReadVariableOp2d
0Online/conv_stack/conv2d_4/Conv2D/ReadVariableOp0Online/conv_stack/conv2d_4/Conv2D/ReadVariableOp2j
3Online/conv_stack_1/conv2d_5/BiasAdd/ReadVariableOp3Online/conv_stack_1/conv2d_5/BiasAdd/ReadVariableOp2h
2Online/conv_stack_1/conv2d_5/Conv2D/ReadVariableOp2Online/conv_stack_1/conv2d_5/Conv2D/ReadVariableOp2j
3Online/conv_stack_1/conv2d_6/BiasAdd/ReadVariableOp3Online/conv_stack_1/conv2d_6/BiasAdd/ReadVariableOp2h
2Online/conv_stack_1/conv2d_6/Conv2D/ReadVariableOp2Online/conv_stack_1/conv2d_6/Conv2D/ReadVariableOp2j
3Online/conv_stack_2/conv2d_7/BiasAdd/ReadVariableOp3Online/conv_stack_2/conv2d_7/BiasAdd/ReadVariableOp2h
2Online/conv_stack_2/conv2d_7/Conv2D/ReadVariableOp2Online/conv_stack_2/conv2d_7/Conv2D/ReadVariableOp2j
3Online/conv_stack_2/conv2d_8/BiasAdd/ReadVariableOp3Online/conv_stack_2/conv2d_8/BiasAdd/ReadVariableOp2h
2Online/conv_stack_2/conv2d_8/Conv2D/ReadVariableOp2Online/conv_stack_2/conv2d_8/Conv2D/ReadVariableOp2j
3Online/conv_stack_2/conv2d_9/BiasAdd/ReadVariableOp3Online/conv_stack_2/conv2d_9/BiasAdd/ReadVariableOp2h
2Online/conv_stack_2/conv2d_9/Conv2D/ReadVariableOp2Online/conv_stack_2/conv2d_9/Conv2D/ReadVariableOp2l
4Online/conv_stack_3/conv2d_10/BiasAdd/ReadVariableOp4Online/conv_stack_3/conv2d_10/BiasAdd/ReadVariableOp2j
3Online/conv_stack_3/conv2d_10/Conv2D/ReadVariableOp3Online/conv_stack_3/conv2d_10/Conv2D/ReadVariableOp2l
4Online/conv_stack_3/conv2d_11/BiasAdd/ReadVariableOp4Online/conv_stack_3/conv2d_11/BiasAdd/ReadVariableOp2j
3Online/conv_stack_3/conv2d_11/Conv2D/ReadVariableOp3Online/conv_stack_3/conv2d_11/Conv2D/ReadVariableOp2l
4Online/conv_stack_4/conv2d_12/BiasAdd/ReadVariableOp4Online/conv_stack_4/conv2d_12/BiasAdd/ReadVariableOp2j
3Online/conv_stack_4/conv2d_12/Conv2D/ReadVariableOp3Online/conv_stack_4/conv2d_12/Conv2D/ReadVariableOp2l
4Online/conv_stack_4/conv2d_13/BiasAdd/ReadVariableOp4Online/conv_stack_4/conv2d_13/BiasAdd/ReadVariableOp2j
3Online/conv_stack_4/conv2d_13/Conv2D/ReadVariableOp3Online/conv_stack_4/conv2d_13/Conv2D/ReadVariableOp2l
4Online/conv_stack_4/conv2d_14/BiasAdd/ReadVariableOp4Online/conv_stack_4/conv2d_14/BiasAdd/ReadVariableOp2j
3Online/conv_stack_4/conv2d_14/Conv2D/ReadVariableOp3Online/conv_stack_4/conv2d_14/Conv2D/ReadVariableOp2l
4Online/conv_stack_5/conv2d_15/BiasAdd/ReadVariableOp4Online/conv_stack_5/conv2d_15/BiasAdd/ReadVariableOp2j
3Online/conv_stack_5/conv2d_15/Conv2D/ReadVariableOp3Online/conv_stack_5/conv2d_15/Conv2D/ReadVariableOp2l
4Online/conv_stack_5/conv2d_16/BiasAdd/ReadVariableOp4Online/conv_stack_5/conv2d_16/BiasAdd/ReadVariableOp2j
3Online/conv_stack_5/conv2d_16/Conv2D/ReadVariableOp3Online/conv_stack_5/conv2d_16/Conv2D/ReadVariableOp2J
#Online/dense/BiasAdd/ReadVariableOp#Online/dense/BiasAdd/ReadVariableOp2H
"Online/dense/MatMul/ReadVariableOp"Online/dense/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������TT
!
_user_specified_name	input_1
�
a
E__inference_leaky_re_lu_layer_call_and_return_conditional_losses_2440

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������TT0*
alpha%���>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������TT02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������TT0:W S
/
_output_shapes
:���������TT0
 
_user_specified_nameinputs
�
�
F__inference_conv_stack_3_layer_call_and_return_conditional_losses_2576

inputs`
Dconv2d_10_conv2d_readvariableop_online_conv_stack_3_conv2d_10_kernel:��R
Cconv2d_10_biasadd_readvariableop_online_conv_stack_3_conv2d_10_bias:	�`
Dconv2d_11_conv2d_readvariableop_online_conv_stack_3_conv2d_11_kernel:��R
Cconv2d_11_biasadd_readvariableop_online_conv_stack_3_conv2d_11_bias:	�
identity�� conv2d_10/BiasAdd/ReadVariableOp�conv2d_10/Conv2D/ReadVariableOp� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp�
conv2d_10/Conv2D/ReadVariableOpReadVariableOpDconv2d_10_conv2d_readvariableop_online_conv_stack_3_conv2d_10_kernel*(
_output_shapes
:��*
dtype02!
conv2d_10/Conv2D/ReadVariableOp�
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_10/Conv2D�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOpCconv2d_10_biasadd_readvariableop_online_conv_stack_3_conv2d_10_bias*
_output_shapes	
:�*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp�
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_10/BiasAdd�
leaky_re_lu_8/LeakyRelu	LeakyReluconv2d_10/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_8/LeakyRelu�
conv2d_11/Conv2D/ReadVariableOpReadVariableOpDconv2d_11_conv2d_readvariableop_online_conv_stack_3_conv2d_11_kernel*(
_output_shapes
:��*
dtype02!
conv2d_11/Conv2D/ReadVariableOp�
conv2d_11/Conv2DConv2D%leaky_re_lu_8/LeakyRelu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_11/Conv2D�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOpCconv2d_11_biasadd_readvariableop_online_conv_stack_3_conv2d_11_bias*
_output_shapes	
:�*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp�
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_11/BiasAdd~
	add_3/addAddV2conv2d_11/BiasAdd:output:0inputs*
T0*0
_output_shapes
:����������2
	add_3/add�
leaky_re_lu_9/LeakyRelu	LeakyReluadd_3/add:z:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_9/LeakyRelu�
IdentityIdentity%leaky_re_lu_9/LeakyRelu:activations:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
?__inference_dense_layer_call_and_return_conditional_losses_3334

inputs<
)matmul_readvariableop_online_dense_kernel:	�6
(biasadd_readvariableop_online_dense_bias:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp)matmul_readvariableop_online_dense_kernel*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_online_dense_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_dense_layer_call_fn_3341

inputs&
online_dense_kernel:	�
online_dense_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsonline_dense_kernelonline_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_26522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
n
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_2411

inputs
identity�
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
%__inference_conv2d_layer_call_fn_3092

inputs.
online_conv2d_kernel:0 
online_conv2d_bias:0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsonline_conv2d_kernelonline_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������TT0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_24312
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������TT02

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������TT: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������TT
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2392

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
��
�
 __inference__traced_restore_3590
file_prefix?
%assignvariableop_online_conv2d_kernel:03
%assignvariableop_1_online_conv2d_bias:0C
)assignvariableop_2_online_conv2d_1_kernel:005
'assignvariableop_3_online_conv2d_1_bias:0N
4assignvariableop_4_online_conv_stack_conv2d_2_kernel:0H@
2assignvariableop_5_online_conv_stack_conv2d_2_bias:HN
4assignvariableop_6_online_conv_stack_conv2d_3_kernel:HH@
2assignvariableop_7_online_conv_stack_conv2d_3_bias:HN
4assignvariableop_8_online_conv_stack_conv2d_4_kernel:0H@
2assignvariableop_9_online_conv_stack_conv2d_4_bias:HQ
7assignvariableop_10_online_conv_stack_1_conv2d_5_kernel:HHC
5assignvariableop_11_online_conv_stack_1_conv2d_5_bias:HQ
7assignvariableop_12_online_conv_stack_1_conv2d_6_kernel:HHC
5assignvariableop_13_online_conv_stack_1_conv2d_6_bias:HR
7assignvariableop_14_online_conv_stack_2_conv2d_7_kernel:H�D
5assignvariableop_15_online_conv_stack_2_conv2d_7_bias:	�S
7assignvariableop_16_online_conv_stack_2_conv2d_8_kernel:��D
5assignvariableop_17_online_conv_stack_2_conv2d_8_bias:	�R
7assignvariableop_18_online_conv_stack_2_conv2d_9_kernel:H�D
5assignvariableop_19_online_conv_stack_2_conv2d_9_bias:	�T
8assignvariableop_20_online_conv_stack_3_conv2d_10_kernel:��E
6assignvariableop_21_online_conv_stack_3_conv2d_10_bias:	�T
8assignvariableop_22_online_conv_stack_3_conv2d_11_kernel:��E
6assignvariableop_23_online_conv_stack_3_conv2d_11_bias:	�T
8assignvariableop_24_online_conv_stack_4_conv2d_12_kernel:��E
6assignvariableop_25_online_conv_stack_4_conv2d_12_bias:	�T
8assignvariableop_26_online_conv_stack_4_conv2d_13_kernel:��E
6assignvariableop_27_online_conv_stack_4_conv2d_13_bias:	�T
8assignvariableop_28_online_conv_stack_4_conv2d_14_kernel:��E
6assignvariableop_29_online_conv_stack_4_conv2d_14_bias:	�T
8assignvariableop_30_online_conv_stack_5_conv2d_15_kernel:��E
6assignvariableop_31_online_conv_stack_5_conv2d_15_bias:	�T
8assignvariableop_32_online_conv_stack_5_conv2d_16_kernel:��E
6assignvariableop_33_online_conv_stack_5_conv2d_16_bias:	�:
'assignvariableop_34_online_dense_kernel:	�3
%assignvariableop_35_online_dense_bias:
identity_37��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp%assignvariableop_online_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp%assignvariableop_1_online_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp)assignvariableop_2_online_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp'assignvariableop_3_online_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp4assignvariableop_4_online_conv_stack_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp2assignvariableop_5_online_conv_stack_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp4assignvariableop_6_online_conv_stack_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp2assignvariableop_7_online_conv_stack_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp4assignvariableop_8_online_conv_stack_conv2d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp2assignvariableop_9_online_conv_stack_conv2d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp7assignvariableop_10_online_conv_stack_1_conv2d_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp5assignvariableop_11_online_conv_stack_1_conv2d_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp7assignvariableop_12_online_conv_stack_1_conv2d_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp5assignvariableop_13_online_conv_stack_1_conv2d_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp7assignvariableop_14_online_conv_stack_2_conv2d_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp5assignvariableop_15_online_conv_stack_2_conv2d_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp7assignvariableop_16_online_conv_stack_2_conv2d_8_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp5assignvariableop_17_online_conv_stack_2_conv2d_8_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp7assignvariableop_18_online_conv_stack_2_conv2d_9_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp5assignvariableop_19_online_conv_stack_2_conv2d_9_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp8assignvariableop_20_online_conv_stack_3_conv2d_10_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp6assignvariableop_21_online_conv_stack_3_conv2d_10_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp8assignvariableop_22_online_conv_stack_3_conv2d_11_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp6assignvariableop_23_online_conv_stack_3_conv2d_11_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp8assignvariableop_24_online_conv_stack_4_conv2d_12_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp6assignvariableop_25_online_conv_stack_4_conv2d_12_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp8assignvariableop_26_online_conv_stack_4_conv2d_13_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp6assignvariableop_27_online_conv_stack_4_conv2d_13_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp8assignvariableop_28_online_conv_stack_4_conv2d_14_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp6assignvariableop_29_online_conv_stack_4_conv2d_14_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp8assignvariableop_30_online_conv_stack_5_conv2d_15_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp6assignvariableop_31_online_conv_stack_5_conv2d_15_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp8assignvariableop_32_online_conv_stack_5_conv2d_16_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp6assignvariableop_33_online_conv_stack_5_conv2d_16_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_online_dense_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp%assignvariableop_35_online_dense_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_359
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_36�
Identity_37IdentityIdentity_36:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_37"#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_35AssignVariableOp_352(
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
�
J
.__inference_max_pooling2d_2_layer_call_fn_2378

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_23752
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
'__inference_conv2d_1_layer_call_fn_3119

inputs0
online_conv2d_1_kernel:00"
online_conv2d_1_bias:0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsonline_conv2d_1_kernelonline_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������TT0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_24522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������TT02

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������TT0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������TT0
 
_user_specified_nameinputs
�
c
G__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_3124

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������))0*
alpha%���>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������))02

Identity"
identityIdentity:output:0*.
_input_shapes
:���������))0:W S
/
_output_shapes
:���������))0
 
_user_specified_nameinputs
�
S
7__inference_global_average_pooling2d_layer_call_fn_2414

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_24112
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
+__inference_conv_stack_5_layer_call_fn_3324

inputs@
$online_conv_stack_5_conv2d_15_kernel:��1
"online_conv_stack_5_conv2d_15_bias:	�@
$online_conv_stack_5_conv2d_16_kernel:��1
"online_conv_stack_5_conv2d_16_bias:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs$online_conv_stack_5_conv2d_15_kernel"online_conv_stack_5_conv2d_15_bias$online_conv_stack_5_conv2d_16_kernel"online_conv_stack_5_conv2d_16_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_5_layer_call_and_return_conditional_losses_26352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
?__inference_dense_layer_call_and_return_conditional_losses_2652

inputs<
)matmul_readvariableop_online_dense_kernel:	�6
(biasadd_readvariableop_online_dense_bias:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp)matmul_readvariableop_online_dense_kernel*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_online_dense_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_conv_stack_5_layer_call_and_return_conditional_losses_2635

inputs`
Dconv2d_15_conv2d_readvariableop_online_conv_stack_5_conv2d_15_kernel:��R
Cconv2d_15_biasadd_readvariableop_online_conv_stack_5_conv2d_15_bias:	�`
Dconv2d_16_conv2d_readvariableop_online_conv_stack_5_conv2d_16_kernel:��R
Cconv2d_16_biasadd_readvariableop_online_conv_stack_5_conv2d_16_bias:	�
identity�� conv2d_15/BiasAdd/ReadVariableOp�conv2d_15/Conv2D/ReadVariableOp� conv2d_16/BiasAdd/ReadVariableOp�conv2d_16/Conv2D/ReadVariableOp�
conv2d_15/Conv2D/ReadVariableOpReadVariableOpDconv2d_15_conv2d_readvariableop_online_conv_stack_5_conv2d_15_kernel*(
_output_shapes
:��*
dtype02!
conv2d_15/Conv2D/ReadVariableOp�
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_15/Conv2D�
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOpCconv2d_15_biasadd_readvariableop_online_conv_stack_5_conv2d_15_bias*
_output_shapes	
:�*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp�
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_15/BiasAdd�
leaky_re_lu_12/LeakyRelu	LeakyReluconv2d_15/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_12/LeakyRelu�
conv2d_16/Conv2D/ReadVariableOpReadVariableOpDconv2d_16_conv2d_readvariableop_online_conv_stack_5_conv2d_16_kernel*(
_output_shapes
:��*
dtype02!
conv2d_16/Conv2D/ReadVariableOp�
conv2d_16/Conv2DConv2D&leaky_re_lu_12/LeakyRelu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_16/Conv2D�
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOpCconv2d_16_biasadd_readvariableop_online_conv_stack_5_conv2d_16_bias*
_output_shapes	
:�*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp�
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_16/BiasAdd~
	add_5/addAddV2conv2d_16/BiasAdd:output:0inputs*
T0*0
_output_shapes
:����������2
	add_5/add�
leaky_re_lu_13/LeakyRelu	LeakyReluadd_5/add:z:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_13/LeakyRelu�
IdentityIdentity&leaky_re_lu_13/LeakyRelu:activations:0!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : 2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
H
,__inference_max_pooling2d_layer_call_fn_2344

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_23412
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
c
G__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_2462

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:���������))0*
alpha%���>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:���������))02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������))0:W S
/
_output_shapes
:���������))0
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2350

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
+__inference_conv_stack_2_layer_call_fn_3231

inputs>
#online_conv_stack_2_conv2d_7_kernel:H�0
!online_conv_stack_2_conv2d_7_bias:	�?
#online_conv_stack_2_conv2d_8_kernel:��0
!online_conv_stack_2_conv2d_8_bias:	�>
#online_conv_stack_2_conv2d_9_kernel:H�0
!online_conv_stack_2_conv2d_9_bias:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs#online_conv_stack_2_conv2d_7_kernel!online_conv_stack_2_conv2d_7_bias#online_conv_stack_2_conv2d_8_kernel!online_conv_stack_2_conv2d_8_bias#online_conv_stack_2_conv2d_9_kernel!online_conv_stack_2_conv2d_9_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_2_layer_call_and_return_conditional_losses_25492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������H: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
�
+__inference_conv_stack_4_layer_call_fn_3296

inputs@
$online_conv_stack_4_conv2d_12_kernel:��1
"online_conv_stack_4_conv2d_12_bias:	�@
$online_conv_stack_4_conv2d_13_kernel:��1
"online_conv_stack_4_conv2d_13_bias:	�@
$online_conv_stack_4_conv2d_14_kernel:��1
"online_conv_stack_4_conv2d_14_bias:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs$online_conv_stack_4_conv2d_12_kernel"online_conv_stack_4_conv2d_12_bias$online_conv_stack_4_conv2d_13_kernel"online_conv_stack_4_conv2d_13_bias$online_conv_stack_4_conv2d_14_kernel"online_conv_stack_4_conv2d_14_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_4_layer_call_and_return_conditional_losses_26082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
"__inference_signature_wrapper_3075
input_1.
online_conv2d_kernel:0 
online_conv2d_bias:00
online_conv2d_1_kernel:00"
online_conv2d_1_bias:0;
!online_conv_stack_conv2d_2_kernel:0H-
online_conv_stack_conv2d_2_bias:H;
!online_conv_stack_conv2d_3_kernel:HH-
online_conv_stack_conv2d_3_bias:H;
!online_conv_stack_conv2d_4_kernel:0H-
online_conv_stack_conv2d_4_bias:H=
#online_conv_stack_1_conv2d_5_kernel:HH/
!online_conv_stack_1_conv2d_5_bias:H=
#online_conv_stack_1_conv2d_6_kernel:HH/
!online_conv_stack_1_conv2d_6_bias:H>
#online_conv_stack_2_conv2d_7_kernel:H�0
!online_conv_stack_2_conv2d_7_bias:	�?
#online_conv_stack_2_conv2d_8_kernel:��0
!online_conv_stack_2_conv2d_8_bias:	�>
#online_conv_stack_2_conv2d_9_kernel:H�0
!online_conv_stack_2_conv2d_9_bias:	�@
$online_conv_stack_3_conv2d_10_kernel:��1
"online_conv_stack_3_conv2d_10_bias:	�@
$online_conv_stack_3_conv2d_11_kernel:��1
"online_conv_stack_3_conv2d_11_bias:	�@
$online_conv_stack_4_conv2d_12_kernel:��1
"online_conv_stack_4_conv2d_12_bias:	�@
$online_conv_stack_4_conv2d_13_kernel:��1
"online_conv_stack_4_conv2d_13_bias:	�@
$online_conv_stack_4_conv2d_14_kernel:��1
"online_conv_stack_4_conv2d_14_bias:	�@
$online_conv_stack_5_conv2d_15_kernel:��1
"online_conv_stack_5_conv2d_15_bias:	�@
$online_conv_stack_5_conv2d_16_kernel:��1
"online_conv_stack_5_conv2d_16_bias:	�&
online_dense_kernel:	�
online_dense_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1online_conv2d_kernelonline_conv2d_biasonline_conv2d_1_kernelonline_conv2d_1_bias!online_conv_stack_conv2d_2_kernelonline_conv_stack_conv2d_2_bias!online_conv_stack_conv2d_3_kernelonline_conv_stack_conv2d_3_bias!online_conv_stack_conv2d_4_kernelonline_conv_stack_conv2d_4_bias#online_conv_stack_1_conv2d_5_kernel!online_conv_stack_1_conv2d_5_bias#online_conv_stack_1_conv2d_6_kernel!online_conv_stack_1_conv2d_6_bias#online_conv_stack_2_conv2d_7_kernel!online_conv_stack_2_conv2d_7_bias#online_conv_stack_2_conv2d_8_kernel!online_conv_stack_2_conv2d_8_bias#online_conv_stack_2_conv2d_9_kernel!online_conv_stack_2_conv2d_9_bias$online_conv_stack_3_conv2d_10_kernel"online_conv_stack_3_conv2d_10_bias$online_conv_stack_3_conv2d_11_kernel"online_conv_stack_3_conv2d_11_bias$online_conv_stack_4_conv2d_12_kernel"online_conv_stack_4_conv2d_12_bias$online_conv_stack_4_conv2d_13_kernel"online_conv_stack_4_conv2d_13_bias$online_conv_stack_4_conv2d_14_kernel"online_conv_stack_4_conv2d_14_bias$online_conv_stack_5_conv2d_15_kernel"online_conv_stack_5_conv2d_15_bias$online_conv_stack_5_conv2d_16_kernel"online_conv_stack_5_conv2d_16_biasonline_dense_kernelonline_dense_bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__wrapped_model_23272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*v
_input_shapese
c:���������TT: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������TT
!
_user_specified_name	input_1
�&
�
F__inference_conv_stack_4_layer_call_and_return_conditional_losses_2608

inputs`
Dconv2d_12_conv2d_readvariableop_online_conv_stack_4_conv2d_12_kernel:��R
Cconv2d_12_biasadd_readvariableop_online_conv_stack_4_conv2d_12_bias:	�`
Dconv2d_13_conv2d_readvariableop_online_conv_stack_4_conv2d_13_kernel:��R
Cconv2d_13_biasadd_readvariableop_online_conv_stack_4_conv2d_13_bias:	�`
Dconv2d_14_conv2d_readvariableop_online_conv_stack_4_conv2d_14_kernel:��R
Cconv2d_14_biasadd_readvariableop_online_conv_stack_4_conv2d_14_bias:	�
identity�� conv2d_12/BiasAdd/ReadVariableOp�conv2d_12/Conv2D/ReadVariableOp� conv2d_13/BiasAdd/ReadVariableOp�conv2d_13/Conv2D/ReadVariableOp� conv2d_14/BiasAdd/ReadVariableOp�conv2d_14/Conv2D/ReadVariableOp�
conv2d_12/Conv2D/ReadVariableOpReadVariableOpDconv2d_12_conv2d_readvariableop_online_conv_stack_4_conv2d_12_kernel*(
_output_shapes
:��*
dtype02!
conv2d_12/Conv2D/ReadVariableOp�
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_12/Conv2D�
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOpCconv2d_12_biasadd_readvariableop_online_conv_stack_4_conv2d_12_bias*
_output_shapes	
:�*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp�
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_12/BiasAdd�
leaky_re_lu_10/LeakyRelu	LeakyReluconv2d_12/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_10/LeakyRelu�
conv2d_13/Conv2D/ReadVariableOpReadVariableOpDconv2d_13_conv2d_readvariableop_online_conv_stack_4_conv2d_13_kernel*(
_output_shapes
:��*
dtype02!
conv2d_13/Conv2D/ReadVariableOp�
conv2d_13/Conv2DConv2D&leaky_re_lu_10/LeakyRelu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_13/Conv2D�
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOpCconv2d_13_biasadd_readvariableop_online_conv_stack_4_conv2d_13_bias*
_output_shapes	
:�*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp�
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_13/BiasAdd�
max_pooling2d_3/MaxPoolMaxPoolconv2d_13/BiasAdd:output:0*0
_output_shapes
:����������*
ksize
*
paddingSAME*
strides
2
max_pooling2d_3/MaxPool�
conv2d_14/Conv2D/ReadVariableOpReadVariableOpDconv2d_14_conv2d_readvariableop_online_conv_stack_4_conv2d_14_kernel*(
_output_shapes
:��*
dtype02!
conv2d_14/Conv2D/ReadVariableOp�
conv2d_14/Conv2DConv2Dinputs'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_14/Conv2D�
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOpCconv2d_14_biasadd_readvariableop_online_conv_stack_4_conv2d_14_bias*
_output_shapes	
:�*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp�
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_14/BiasAdd�
	add_4/addAddV2 max_pooling2d_3/MaxPool:output:0conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
	add_4/add�
leaky_re_lu_11/LeakyRelu	LeakyReluadd_4/add:z:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_11/LeakyRelu�
IdentityIdentity&leaky_re_lu_11/LeakyRelu:activations:0!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_leaky_re_lu_layer_call_fn_3102

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������TT0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_leaky_re_lu_layer_call_and_return_conditional_losses_24402
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������TT02

Identity"
identityIdentity:output:0*.
_input_shapes
:���������TT0:W S
/
_output_shapes
:���������TT0
 
_user_specified_nameinputs
�Z
�
@__inference_Online_layer_call_and_return_conditional_losses_2657
input_15
conv2d_online_conv2d_kernel:0'
conv2d_online_conv2d_bias:09
conv2d_1_online_conv2d_1_kernel:00+
conv2d_1_online_conv2d_1_bias:0F
,conv_stack_online_conv_stack_conv2d_2_kernel:0H8
*conv_stack_online_conv_stack_conv2d_2_bias:HF
,conv_stack_online_conv_stack_conv2d_3_kernel:HH8
*conv_stack_online_conv_stack_conv2d_3_bias:HF
,conv_stack_online_conv_stack_conv2d_4_kernel:0H8
*conv_stack_online_conv_stack_conv2d_4_bias:HJ
0conv_stack_1_online_conv_stack_1_conv2d_5_kernel:HH<
.conv_stack_1_online_conv_stack_1_conv2d_5_bias:HJ
0conv_stack_1_online_conv_stack_1_conv2d_6_kernel:HH<
.conv_stack_1_online_conv_stack_1_conv2d_6_bias:HK
0conv_stack_2_online_conv_stack_2_conv2d_7_kernel:H�=
.conv_stack_2_online_conv_stack_2_conv2d_7_bias:	�L
0conv_stack_2_online_conv_stack_2_conv2d_8_kernel:��=
.conv_stack_2_online_conv_stack_2_conv2d_8_bias:	�K
0conv_stack_2_online_conv_stack_2_conv2d_9_kernel:H�=
.conv_stack_2_online_conv_stack_2_conv2d_9_bias:	�M
1conv_stack_3_online_conv_stack_3_conv2d_10_kernel:��>
/conv_stack_3_online_conv_stack_3_conv2d_10_bias:	�M
1conv_stack_3_online_conv_stack_3_conv2d_11_kernel:��>
/conv_stack_3_online_conv_stack_3_conv2d_11_bias:	�M
1conv_stack_4_online_conv_stack_4_conv2d_12_kernel:��>
/conv_stack_4_online_conv_stack_4_conv2d_12_bias:	�M
1conv_stack_4_online_conv_stack_4_conv2d_13_kernel:��>
/conv_stack_4_online_conv_stack_4_conv2d_13_bias:	�M
1conv_stack_4_online_conv_stack_4_conv2d_14_kernel:��>
/conv_stack_4_online_conv_stack_4_conv2d_14_bias:	�M
1conv_stack_5_online_conv_stack_5_conv2d_15_kernel:��>
/conv_stack_5_online_conv_stack_5_conv2d_15_bias:	�M
1conv_stack_5_online_conv_stack_5_conv2d_16_kernel:��>
/conv_stack_5_online_conv_stack_5_conv2d_16_bias:	�,
dense_online_dense_kernel:	�%
dense_online_dense_bias:
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�"conv_stack/StatefulPartitionedCall�$conv_stack_1/StatefulPartitionedCall�$conv_stack_2/StatefulPartitionedCall�$conv_stack_3/StatefulPartitionedCall�$conv_stack_4/StatefulPartitionedCall�$conv_stack_5/StatefulPartitionedCall�dense/StatefulPartitionedCallf
CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:���������TT2
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
:���������TT2	
truediv�
conv2d/StatefulPartitionedCallStatefulPartitionedCalltruediv:z:0conv2d_online_conv2d_kernelconv2d_online_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������TT0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_24312 
conv2d/StatefulPartitionedCall�
leaky_re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������TT0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_leaky_re_lu_layer_call_and_return_conditional_losses_24402
leaky_re_lu/PartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0conv2d_1_online_conv2d_1_kernelconv2d_1_online_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������TT0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_24522"
 conv2d_1/StatefulPartitionedCall�
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������))0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_23412
max_pooling2d/PartitionedCall�
leaky_re_lu_1/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������))0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_24622
leaky_re_lu_1/PartitionedCall�
"conv_stack/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0,conv_stack_online_conv_stack_conv2d_2_kernel*conv_stack_online_conv_stack_conv2d_2_bias,conv_stack_online_conv_stack_conv2d_3_kernel*conv_stack_online_conv_stack_conv2d_3_bias,conv_stack_online_conv_stack_conv2d_4_kernel*conv_stack_online_conv_stack_conv2d_4_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv_stack_layer_call_and_return_conditional_losses_24902$
"conv_stack/StatefulPartitionedCall�
$conv_stack_1/StatefulPartitionedCallStatefulPartitionedCall+conv_stack/StatefulPartitionedCall:output:00conv_stack_1_online_conv_stack_1_conv2d_5_kernel.conv_stack_1_online_conv_stack_1_conv2d_5_bias0conv_stack_1_online_conv_stack_1_conv2d_6_kernel.conv_stack_1_online_conv_stack_1_conv2d_6_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_1_layer_call_and_return_conditional_losses_25172&
$conv_stack_1/StatefulPartitionedCall�
$conv_stack_2/StatefulPartitionedCallStatefulPartitionedCall-conv_stack_1/StatefulPartitionedCall:output:00conv_stack_2_online_conv_stack_2_conv2d_7_kernel.conv_stack_2_online_conv_stack_2_conv2d_7_bias0conv_stack_2_online_conv_stack_2_conv2d_8_kernel.conv_stack_2_online_conv_stack_2_conv2d_8_bias0conv_stack_2_online_conv_stack_2_conv2d_9_kernel.conv_stack_2_online_conv_stack_2_conv2d_9_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_2_layer_call_and_return_conditional_losses_25492&
$conv_stack_2/StatefulPartitionedCall�
$conv_stack_3/StatefulPartitionedCallStatefulPartitionedCall-conv_stack_2/StatefulPartitionedCall:output:01conv_stack_3_online_conv_stack_3_conv2d_10_kernel/conv_stack_3_online_conv_stack_3_conv2d_10_bias1conv_stack_3_online_conv_stack_3_conv2d_11_kernel/conv_stack_3_online_conv_stack_3_conv2d_11_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_3_layer_call_and_return_conditional_losses_25762&
$conv_stack_3/StatefulPartitionedCall�
$conv_stack_4/StatefulPartitionedCallStatefulPartitionedCall-conv_stack_3/StatefulPartitionedCall:output:01conv_stack_4_online_conv_stack_4_conv2d_12_kernel/conv_stack_4_online_conv_stack_4_conv2d_12_bias1conv_stack_4_online_conv_stack_4_conv2d_13_kernel/conv_stack_4_online_conv_stack_4_conv2d_13_bias1conv_stack_4_online_conv_stack_4_conv2d_14_kernel/conv_stack_4_online_conv_stack_4_conv2d_14_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_4_layer_call_and_return_conditional_losses_26082&
$conv_stack_4/StatefulPartitionedCall�
$conv_stack_5/StatefulPartitionedCallStatefulPartitionedCall-conv_stack_4/StatefulPartitionedCall:output:01conv_stack_5_online_conv_stack_5_conv2d_15_kernel/conv_stack_5_online_conv_stack_5_conv2d_15_bias1conv_stack_5_online_conv_stack_5_conv2d_16_kernel/conv_stack_5_online_conv_stack_5_conv2d_16_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_5_layer_call_and_return_conditional_losses_26352&
$conv_stack_5/StatefulPartitionedCall�
(global_average_pooling2d/PartitionedCallPartitionedCall-conv_stack_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_24112*
(global_average_pooling2d/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_online_dense_kerneldense_online_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_26522
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall#^conv_stack/StatefulPartitionedCall%^conv_stack_1/StatefulPartitionedCall%^conv_stack_2/StatefulPartitionedCall%^conv_stack_3/StatefulPartitionedCall%^conv_stack_4/StatefulPartitionedCall%^conv_stack_5/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*v
_input_shapese
c:���������TT: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2H
"conv_stack/StatefulPartitionedCall"conv_stack/StatefulPartitionedCall2L
$conv_stack_1/StatefulPartitionedCall$conv_stack_1/StatefulPartitionedCall2L
$conv_stack_2/StatefulPartitionedCall$conv_stack_2/StatefulPartitionedCall2L
$conv_stack_3/StatefulPartitionedCall$conv_stack_3/StatefulPartitionedCall2L
$conv_stack_4/StatefulPartitionedCall$conv_stack_4/StatefulPartitionedCall2L
$conv_stack_5/StatefulPartitionedCall$conv_stack_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:X T
/
_output_shapes
:���������TT
!
_user_specified_name	input_1
�
�
F__inference_conv_stack_3_layer_call_and_return_conditional_losses_3250

inputs`
Dconv2d_10_conv2d_readvariableop_online_conv_stack_3_conv2d_10_kernel:��R
Cconv2d_10_biasadd_readvariableop_online_conv_stack_3_conv2d_10_bias:	�`
Dconv2d_11_conv2d_readvariableop_online_conv_stack_3_conv2d_11_kernel:��R
Cconv2d_11_biasadd_readvariableop_online_conv_stack_3_conv2d_11_bias:	�
identity�� conv2d_10/BiasAdd/ReadVariableOp�conv2d_10/Conv2D/ReadVariableOp� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp�
conv2d_10/Conv2D/ReadVariableOpReadVariableOpDconv2d_10_conv2d_readvariableop_online_conv_stack_3_conv2d_10_kernel*(
_output_shapes
:��*
dtype02!
conv2d_10/Conv2D/ReadVariableOp�
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_10/Conv2D�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOpCconv2d_10_biasadd_readvariableop_online_conv_stack_3_conv2d_10_bias*
_output_shapes	
:�*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp�
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_10/BiasAdd�
leaky_re_lu_8/LeakyRelu	LeakyReluconv2d_10/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_8/LeakyRelu�
conv2d_11/Conv2D/ReadVariableOpReadVariableOpDconv2d_11_conv2d_readvariableop_online_conv_stack_3_conv2d_11_kernel*(
_output_shapes
:��*
dtype02!
conv2d_11/Conv2D/ReadVariableOp�
conv2d_11/Conv2DConv2D%leaky_re_lu_8/LeakyRelu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_11/Conv2D�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOpCconv2d_11_biasadd_readvariableop_online_conv_stack_3_conv2d_11_bias*
_output_shapes	
:�*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp�
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_11/BiasAdd~
	add_3/addAddV2conv2d_11/BiasAdd:output:0inputs*
T0*0
_output_shapes
:����������2
	add_3/add�
leaky_re_lu_9/LeakyRelu	LeakyReluadd_3/add:z:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_9/LeakyRelu�
IdentityIdentity%leaky_re_lu_9/LeakyRelu:activations:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������: : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
@__inference_conv2d_layer_call_and_return_conditional_losses_2431

inputsD
*conv2d_readvariableop_online_conv2d_kernel:07
)biasadd_readvariableop_online_conv2d_bias:0
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOp*conv2d_readvariableop_online_conv2d_kernel*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT0*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOp)biasadd_readvariableop_online_conv2d_bias*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT02	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������TT02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������TT: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������TT
 
_user_specified_nameinputs
�

�
)__inference_conv_stack_layer_call_fn_3166

inputs;
!online_conv_stack_conv2d_2_kernel:0H-
online_conv_stack_conv2d_2_bias:H;
!online_conv_stack_conv2d_3_kernel:HH-
online_conv_stack_conv2d_3_bias:H;
!online_conv_stack_conv2d_4_kernel:0H-
online_conv_stack_conv2d_4_bias:H
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs!online_conv_stack_conv2d_2_kernelonline_conv_stack_conv2d_2_bias!online_conv_stack_conv2d_3_kernelonline_conv_stack_conv2d_3_bias!online_conv_stack_conv2d_4_kernelonline_conv_stack_conv2d_4_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv_stack_layer_call_and_return_conditional_losses_24902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������H2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������))0: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������))0
 
_user_specified_nameinputs
�%
�
F__inference_conv_stack_2_layer_call_and_return_conditional_losses_2549

inputs]
Bconv2d_7_conv2d_readvariableop_online_conv_stack_2_conv2d_7_kernel:H�P
Aconv2d_7_biasadd_readvariableop_online_conv_stack_2_conv2d_7_bias:	�^
Bconv2d_8_conv2d_readvariableop_online_conv_stack_2_conv2d_8_kernel:��P
Aconv2d_8_biasadd_readvariableop_online_conv_stack_2_conv2d_8_bias:	�]
Bconv2d_9_conv2d_readvariableop_online_conv_stack_2_conv2d_9_kernel:H�P
Aconv2d_9_biasadd_readvariableop_online_conv_stack_2_conv2d_9_bias:	�
identity��conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�
conv2d_7/Conv2D/ReadVariableOpReadVariableOpBconv2d_7_conv2d_readvariableop_online_conv_stack_2_conv2d_7_kernel*'
_output_shapes
:H�*
dtype02 
conv2d_7/Conv2D/ReadVariableOp�
conv2d_7/Conv2DConv2Dinputs&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_7/Conv2D�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOpAconv2d_7_biasadd_readvariableop_online_conv_stack_2_conv2d_7_bias*
_output_shapes	
:�*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_7/BiasAdd�
leaky_re_lu_6/LeakyRelu	LeakyReluconv2d_7/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_6/LeakyRelu�
conv2d_8/Conv2D/ReadVariableOpReadVariableOpBconv2d_8_conv2d_readvariableop_online_conv_stack_2_conv2d_8_kernel*(
_output_shapes
:��*
dtype02 
conv2d_8/Conv2D/ReadVariableOp�
conv2d_8/Conv2DConv2D%leaky_re_lu_6/LeakyRelu:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_8/Conv2D�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOpAconv2d_8_biasadd_readvariableop_online_conv_stack_2_conv2d_8_bias*
_output_shapes	
:�*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_8/BiasAdd�
max_pooling2d_2/MaxPoolMaxPoolconv2d_8/BiasAdd:output:0*0
_output_shapes
:����������*
ksize
*
paddingSAME*
strides
2
max_pooling2d_2/MaxPool�
conv2d_9/Conv2D/ReadVariableOpReadVariableOpBconv2d_9_conv2d_readvariableop_online_conv_stack_2_conv2d_9_kernel*'
_output_shapes
:H�*
dtype02 
conv2d_9/Conv2D/ReadVariableOp�
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_9/Conv2D�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOpAconv2d_9_biasadd_readvariableop_online_conv_stack_2_conv2d_9_bias*
_output_shapes	
:�*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_9/BiasAdd�
	add_2/addAddV2 max_pooling2d_2/MaxPool:output:0conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
	add_2/add�
leaky_re_lu_7/LeakyRelu	LeakyReluadd_2/add:z:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_7/LeakyRelu�
IdentityIdentity%leaky_re_lu_7/LeakyRelu:activations:0 ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������H: : : : : : 2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�

�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2452

inputsF
,conv2d_readvariableop_online_conv2d_1_kernel:009
+biasadd_readvariableop_online_conv2d_1_bias:0
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOp,conv2d_readvariableop_online_conv2d_1_kernel*&
_output_shapes
:00*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT0*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOp+biasadd_readvariableop_online_conv2d_1_bias*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT02	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������TT02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������TT0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������TT0
 
_user_specified_nameinputs
�
�
F__inference_conv_stack_5_layer_call_and_return_conditional_losses_3315

inputs`
Dconv2d_15_conv2d_readvariableop_online_conv_stack_5_conv2d_15_kernel:��R
Cconv2d_15_biasadd_readvariableop_online_conv_stack_5_conv2d_15_bias:	�`
Dconv2d_16_conv2d_readvariableop_online_conv_stack_5_conv2d_16_kernel:��R
Cconv2d_16_biasadd_readvariableop_online_conv_stack_5_conv2d_16_bias:	�
identity�� conv2d_15/BiasAdd/ReadVariableOp�conv2d_15/Conv2D/ReadVariableOp� conv2d_16/BiasAdd/ReadVariableOp�conv2d_16/Conv2D/ReadVariableOp�
conv2d_15/Conv2D/ReadVariableOpReadVariableOpDconv2d_15_conv2d_readvariableop_online_conv_stack_5_conv2d_15_kernel*(
_output_shapes
:��*
dtype02!
conv2d_15/Conv2D/ReadVariableOp�
conv2d_15/Conv2DConv2Dinputs'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_15/Conv2D�
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOpCconv2d_15_biasadd_readvariableop_online_conv_stack_5_conv2d_15_bias*
_output_shapes	
:�*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp�
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_15/BiasAdd�
leaky_re_lu_12/LeakyRelu	LeakyReluconv2d_15/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_12/LeakyRelu�
conv2d_16/Conv2D/ReadVariableOpReadVariableOpDconv2d_16_conv2d_readvariableop_online_conv_stack_5_conv2d_16_kernel*(
_output_shapes
:��*
dtype02!
conv2d_16/Conv2D/ReadVariableOp�
conv2d_16/Conv2DConv2D&leaky_re_lu_12/LeakyRelu:activations:0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_16/Conv2D�
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOpCconv2d_16_biasadd_readvariableop_online_conv_stack_5_conv2d_16_bias*
_output_shapes	
:�*
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp�
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_16/BiasAdd~
	add_5/addAddV2conv2d_16/BiasAdd:output:0inputs*
T0*0
_output_shapes
:����������2
	add_5/add�
leaky_re_lu_13/LeakyRelu	LeakyReluadd_5/add:z:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_13/LeakyRelu�
IdentityIdentity&leaky_re_lu_13/LeakyRelu:activations:0!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������: : : : 2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_conv_stack_1_layer_call_and_return_conditional_losses_2517

inputs\
Bconv2d_5_conv2d_readvariableop_online_conv_stack_1_conv2d_5_kernel:HHO
Aconv2d_5_biasadd_readvariableop_online_conv_stack_1_conv2d_5_bias:H\
Bconv2d_6_conv2d_readvariableop_online_conv_stack_1_conv2d_6_kernel:HHO
Aconv2d_6_biasadd_readvariableop_online_conv_stack_1_conv2d_6_bias:H
identity��conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�
conv2d_5/Conv2D/ReadVariableOpReadVariableOpBconv2d_5_conv2d_readvariableop_online_conv_stack_1_conv2d_5_kernel*&
_output_shapes
:HH*
dtype02 
conv2d_5/Conv2D/ReadVariableOp�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
2
conv2d_5/Conv2D�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOpAconv2d_5_biasadd_readvariableop_online_conv_stack_1_conv2d_5_bias*
_output_shapes
:H*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H2
conv2d_5/BiasAdd�
leaky_re_lu_4/LeakyRelu	LeakyReluconv2d_5/BiasAdd:output:0*/
_output_shapes
:���������H*
alpha%���>2
leaky_re_lu_4/LeakyRelu�
conv2d_6/Conv2D/ReadVariableOpReadVariableOpBconv2d_6_conv2d_readvariableop_online_conv_stack_1_conv2d_6_kernel*&
_output_shapes
:HH*
dtype02 
conv2d_6/Conv2D/ReadVariableOp�
conv2d_6/Conv2DConv2D%leaky_re_lu_4/LeakyRelu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
2
conv2d_6/Conv2D�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOpAconv2d_6_biasadd_readvariableop_online_conv_stack_1_conv2d_6_bias*
_output_shapes
:H*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H2
conv2d_6/BiasAdd|
	add_1/addAddV2conv2d_6/BiasAdd:output:0inputs*
T0*/
_output_shapes
:���������H2
	add_1/add�
leaky_re_lu_5/LeakyRelu	LeakyReluadd_1/add:z:0*/
_output_shapes
:���������H*
alpha%���>2
leaky_re_lu_5/LeakyRelu�
IdentityIdentity%leaky_re_lu_5/LeakyRelu:activations:0 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������H: : : : 2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
J
.__inference_max_pooling2d_1_layer_call_fn_2361

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23582
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2384

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_2333

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2375

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�Q
�
__inference__traced_save_3472
file_prefix3
/savev2_online_conv2d_kernel_read_readvariableop1
-savev2_online_conv2d_bias_read_readvariableop5
1savev2_online_conv2d_1_kernel_read_readvariableop3
/savev2_online_conv2d_1_bias_read_readvariableop@
<savev2_online_conv_stack_conv2d_2_kernel_read_readvariableop>
:savev2_online_conv_stack_conv2d_2_bias_read_readvariableop@
<savev2_online_conv_stack_conv2d_3_kernel_read_readvariableop>
:savev2_online_conv_stack_conv2d_3_bias_read_readvariableop@
<savev2_online_conv_stack_conv2d_4_kernel_read_readvariableop>
:savev2_online_conv_stack_conv2d_4_bias_read_readvariableopB
>savev2_online_conv_stack_1_conv2d_5_kernel_read_readvariableop@
<savev2_online_conv_stack_1_conv2d_5_bias_read_readvariableopB
>savev2_online_conv_stack_1_conv2d_6_kernel_read_readvariableop@
<savev2_online_conv_stack_1_conv2d_6_bias_read_readvariableopB
>savev2_online_conv_stack_2_conv2d_7_kernel_read_readvariableop@
<savev2_online_conv_stack_2_conv2d_7_bias_read_readvariableopB
>savev2_online_conv_stack_2_conv2d_8_kernel_read_readvariableop@
<savev2_online_conv_stack_2_conv2d_8_bias_read_readvariableopB
>savev2_online_conv_stack_2_conv2d_9_kernel_read_readvariableop@
<savev2_online_conv_stack_2_conv2d_9_bias_read_readvariableopC
?savev2_online_conv_stack_3_conv2d_10_kernel_read_readvariableopA
=savev2_online_conv_stack_3_conv2d_10_bias_read_readvariableopC
?savev2_online_conv_stack_3_conv2d_11_kernel_read_readvariableopA
=savev2_online_conv_stack_3_conv2d_11_bias_read_readvariableopC
?savev2_online_conv_stack_4_conv2d_12_kernel_read_readvariableopA
=savev2_online_conv_stack_4_conv2d_12_bias_read_readvariableopC
?savev2_online_conv_stack_4_conv2d_13_kernel_read_readvariableopA
=savev2_online_conv_stack_4_conv2d_13_bias_read_readvariableopC
?savev2_online_conv_stack_4_conv2d_14_kernel_read_readvariableopA
=savev2_online_conv_stack_4_conv2d_14_bias_read_readvariableopC
?savev2_online_conv_stack_5_conv2d_15_kernel_read_readvariableopA
=savev2_online_conv_stack_5_conv2d_15_bias_read_readvariableopC
?savev2_online_conv_stack_5_conv2d_16_kernel_read_readvariableopA
=savev2_online_conv_stack_5_conv2d_16_bias_read_readvariableop2
.savev2_online_dense_kernel_read_readvariableop0
,savev2_online_dense_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_online_conv2d_kernel_read_readvariableop-savev2_online_conv2d_bias_read_readvariableop1savev2_online_conv2d_1_kernel_read_readvariableop/savev2_online_conv2d_1_bias_read_readvariableop<savev2_online_conv_stack_conv2d_2_kernel_read_readvariableop:savev2_online_conv_stack_conv2d_2_bias_read_readvariableop<savev2_online_conv_stack_conv2d_3_kernel_read_readvariableop:savev2_online_conv_stack_conv2d_3_bias_read_readvariableop<savev2_online_conv_stack_conv2d_4_kernel_read_readvariableop:savev2_online_conv_stack_conv2d_4_bias_read_readvariableop>savev2_online_conv_stack_1_conv2d_5_kernel_read_readvariableop<savev2_online_conv_stack_1_conv2d_5_bias_read_readvariableop>savev2_online_conv_stack_1_conv2d_6_kernel_read_readvariableop<savev2_online_conv_stack_1_conv2d_6_bias_read_readvariableop>savev2_online_conv_stack_2_conv2d_7_kernel_read_readvariableop<savev2_online_conv_stack_2_conv2d_7_bias_read_readvariableop>savev2_online_conv_stack_2_conv2d_8_kernel_read_readvariableop<savev2_online_conv_stack_2_conv2d_8_bias_read_readvariableop>savev2_online_conv_stack_2_conv2d_9_kernel_read_readvariableop<savev2_online_conv_stack_2_conv2d_9_bias_read_readvariableop?savev2_online_conv_stack_3_conv2d_10_kernel_read_readvariableop=savev2_online_conv_stack_3_conv2d_10_bias_read_readvariableop?savev2_online_conv_stack_3_conv2d_11_kernel_read_readvariableop=savev2_online_conv_stack_3_conv2d_11_bias_read_readvariableop?savev2_online_conv_stack_4_conv2d_12_kernel_read_readvariableop=savev2_online_conv_stack_4_conv2d_12_bias_read_readvariableop?savev2_online_conv_stack_4_conv2d_13_kernel_read_readvariableop=savev2_online_conv_stack_4_conv2d_13_bias_read_readvariableop?savev2_online_conv_stack_4_conv2d_14_kernel_read_readvariableop=savev2_online_conv_stack_4_conv2d_14_bias_read_readvariableop?savev2_online_conv_stack_5_conv2d_15_kernel_read_readvariableop=savev2_online_conv_stack_5_conv2d_15_bias_read_readvariableop?savev2_online_conv_stack_5_conv2d_16_kernel_read_readvariableop=savev2_online_conv_stack_5_conv2d_16_bias_read_readvariableop.savev2_online_dense_kernel_read_readvariableop,savev2_online_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :0:0:00:0:0H:H:HH:H:0H:H:HH:H:HH:H:H�:�:��:�:H�:�:��:�:��:�:��:�:��:�:��:�:��:�:��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:00: 

_output_shapes
:0:,(
&
_output_shapes
:0H: 

_output_shapes
:H:,(
&
_output_shapes
:HH: 

_output_shapes
:H:,	(
&
_output_shapes
:0H: 


_output_shapes
:H:,(
&
_output_shapes
:HH: 

_output_shapes
:H:,(
&
_output_shapes
:HH: 

_output_shapes
:H:-)
'
_output_shapes
:H�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:-)
'
_output_shapes
:H�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:! 

_output_shapes	
:�:.!*
(
_output_shapes
:��:!"

_output_shapes	
:�:%#!

_output_shapes
:	�: $

_output_shapes
::%

_output_shapes
: 
�Z
�
@__inference_Online_layer_call_and_return_conditional_losses_2716
input_15
conv2d_online_conv2d_kernel:0'
conv2d_online_conv2d_bias:09
conv2d_1_online_conv2d_1_kernel:00+
conv2d_1_online_conv2d_1_bias:0F
,conv_stack_online_conv_stack_conv2d_2_kernel:0H8
*conv_stack_online_conv_stack_conv2d_2_bias:HF
,conv_stack_online_conv_stack_conv2d_3_kernel:HH8
*conv_stack_online_conv_stack_conv2d_3_bias:HF
,conv_stack_online_conv_stack_conv2d_4_kernel:0H8
*conv_stack_online_conv_stack_conv2d_4_bias:HJ
0conv_stack_1_online_conv_stack_1_conv2d_5_kernel:HH<
.conv_stack_1_online_conv_stack_1_conv2d_5_bias:HJ
0conv_stack_1_online_conv_stack_1_conv2d_6_kernel:HH<
.conv_stack_1_online_conv_stack_1_conv2d_6_bias:HK
0conv_stack_2_online_conv_stack_2_conv2d_7_kernel:H�=
.conv_stack_2_online_conv_stack_2_conv2d_7_bias:	�L
0conv_stack_2_online_conv_stack_2_conv2d_8_kernel:��=
.conv_stack_2_online_conv_stack_2_conv2d_8_bias:	�K
0conv_stack_2_online_conv_stack_2_conv2d_9_kernel:H�=
.conv_stack_2_online_conv_stack_2_conv2d_9_bias:	�M
1conv_stack_3_online_conv_stack_3_conv2d_10_kernel:��>
/conv_stack_3_online_conv_stack_3_conv2d_10_bias:	�M
1conv_stack_3_online_conv_stack_3_conv2d_11_kernel:��>
/conv_stack_3_online_conv_stack_3_conv2d_11_bias:	�M
1conv_stack_4_online_conv_stack_4_conv2d_12_kernel:��>
/conv_stack_4_online_conv_stack_4_conv2d_12_bias:	�M
1conv_stack_4_online_conv_stack_4_conv2d_13_kernel:��>
/conv_stack_4_online_conv_stack_4_conv2d_13_bias:	�M
1conv_stack_4_online_conv_stack_4_conv2d_14_kernel:��>
/conv_stack_4_online_conv_stack_4_conv2d_14_bias:	�M
1conv_stack_5_online_conv_stack_5_conv2d_15_kernel:��>
/conv_stack_5_online_conv_stack_5_conv2d_15_bias:	�M
1conv_stack_5_online_conv_stack_5_conv2d_16_kernel:��>
/conv_stack_5_online_conv_stack_5_conv2d_16_bias:	�,
dense_online_dense_kernel:	�%
dense_online_dense_bias:
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�"conv_stack/StatefulPartitionedCall�$conv_stack_1/StatefulPartitionedCall�$conv_stack_2/StatefulPartitionedCall�$conv_stack_3/StatefulPartitionedCall�$conv_stack_4/StatefulPartitionedCall�$conv_stack_5/StatefulPartitionedCall�dense/StatefulPartitionedCallf
CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:���������TT2
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
:���������TT2	
truediv�
conv2d/StatefulPartitionedCallStatefulPartitionedCalltruediv:z:0conv2d_online_conv2d_kernelconv2d_online_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������TT0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_24312 
conv2d/StatefulPartitionedCall�
leaky_re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������TT0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_leaky_re_lu_layer_call_and_return_conditional_losses_24402
leaky_re_lu/PartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0conv2d_1_online_conv2d_1_kernelconv2d_1_online_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������TT0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_24522"
 conv2d_1/StatefulPartitionedCall�
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������))0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_23412
max_pooling2d/PartitionedCall�
leaky_re_lu_1/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������))0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_24622
leaky_re_lu_1/PartitionedCall�
"conv_stack/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0,conv_stack_online_conv_stack_conv2d_2_kernel*conv_stack_online_conv_stack_conv2d_2_bias,conv_stack_online_conv_stack_conv2d_3_kernel*conv_stack_online_conv_stack_conv2d_3_bias,conv_stack_online_conv_stack_conv2d_4_kernel*conv_stack_online_conv_stack_conv2d_4_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv_stack_layer_call_and_return_conditional_losses_24902$
"conv_stack/StatefulPartitionedCall�
$conv_stack_1/StatefulPartitionedCallStatefulPartitionedCall+conv_stack/StatefulPartitionedCall:output:00conv_stack_1_online_conv_stack_1_conv2d_5_kernel.conv_stack_1_online_conv_stack_1_conv2d_5_bias0conv_stack_1_online_conv_stack_1_conv2d_6_kernel.conv_stack_1_online_conv_stack_1_conv2d_6_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_1_layer_call_and_return_conditional_losses_25172&
$conv_stack_1/StatefulPartitionedCall�
$conv_stack_2/StatefulPartitionedCallStatefulPartitionedCall-conv_stack_1/StatefulPartitionedCall:output:00conv_stack_2_online_conv_stack_2_conv2d_7_kernel.conv_stack_2_online_conv_stack_2_conv2d_7_bias0conv_stack_2_online_conv_stack_2_conv2d_8_kernel.conv_stack_2_online_conv_stack_2_conv2d_8_bias0conv_stack_2_online_conv_stack_2_conv2d_9_kernel.conv_stack_2_online_conv_stack_2_conv2d_9_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_2_layer_call_and_return_conditional_losses_25492&
$conv_stack_2/StatefulPartitionedCall�
$conv_stack_3/StatefulPartitionedCallStatefulPartitionedCall-conv_stack_2/StatefulPartitionedCall:output:01conv_stack_3_online_conv_stack_3_conv2d_10_kernel/conv_stack_3_online_conv_stack_3_conv2d_10_bias1conv_stack_3_online_conv_stack_3_conv2d_11_kernel/conv_stack_3_online_conv_stack_3_conv2d_11_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_3_layer_call_and_return_conditional_losses_25762&
$conv_stack_3/StatefulPartitionedCall�
$conv_stack_4/StatefulPartitionedCallStatefulPartitionedCall-conv_stack_3/StatefulPartitionedCall:output:01conv_stack_4_online_conv_stack_4_conv2d_12_kernel/conv_stack_4_online_conv_stack_4_conv2d_12_bias1conv_stack_4_online_conv_stack_4_conv2d_13_kernel/conv_stack_4_online_conv_stack_4_conv2d_13_bias1conv_stack_4_online_conv_stack_4_conv2d_14_kernel/conv_stack_4_online_conv_stack_4_conv2d_14_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_4_layer_call_and_return_conditional_losses_26082&
$conv_stack_4/StatefulPartitionedCall�
$conv_stack_5/StatefulPartitionedCallStatefulPartitionedCall-conv_stack_4/StatefulPartitionedCall:output:01conv_stack_5_online_conv_stack_5_conv2d_15_kernel/conv_stack_5_online_conv_stack_5_conv2d_15_bias1conv_stack_5_online_conv_stack_5_conv2d_16_kernel/conv_stack_5_online_conv_stack_5_conv2d_16_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_5_layer_call_and_return_conditional_losses_26352&
$conv_stack_5/StatefulPartitionedCall�
(global_average_pooling2d/PartitionedCallPartitionedCall-conv_stack_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_24112*
(global_average_pooling2d/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_online_dense_kerneldense_online_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_26522
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall#^conv_stack/StatefulPartitionedCall%^conv_stack_1/StatefulPartitionedCall%^conv_stack_2/StatefulPartitionedCall%^conv_stack_3/StatefulPartitionedCall%^conv_stack_4/StatefulPartitionedCall%^conv_stack_5/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:���������TT: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2H
"conv_stack/StatefulPartitionedCall"conv_stack/StatefulPartitionedCall2L
$conv_stack_1/StatefulPartitionedCall$conv_stack_1/StatefulPartitionedCall2L
$conv_stack_2/StatefulPartitionedCall$conv_stack_2/StatefulPartitionedCall2L
$conv_stack_3/StatefulPartitionedCall$conv_stack_3/StatefulPartitionedCall2L
$conv_stack_4/StatefulPartitionedCall$conv_stack_4/StatefulPartitionedCall2L
$conv_stack_5/StatefulPartitionedCall$conv_stack_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:X T
/
_output_shapes
:���������TT
!
_user_specified_name	input_1
�	
�
+__inference_conv_stack_3_layer_call_fn_3259

inputs@
$online_conv_stack_3_conv2d_10_kernel:��1
"online_conv_stack_3_conv2d_10_bias:	�@
$online_conv_stack_3_conv2d_11_kernel:��1
"online_conv_stack_3_conv2d_11_bias:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs$online_conv_stack_3_conv2d_10_kernel"online_conv_stack_3_conv2d_10_bias$online_conv_stack_3_conv2d_11_kernel"online_conv_stack_3_conv2d_11_bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv_stack_3_layer_call_and_return_conditional_losses_25762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
F__inference_conv_stack_4_layer_call_and_return_conditional_losses_3285

inputs`
Dconv2d_12_conv2d_readvariableop_online_conv_stack_4_conv2d_12_kernel:��R
Cconv2d_12_biasadd_readvariableop_online_conv_stack_4_conv2d_12_bias:	�`
Dconv2d_13_conv2d_readvariableop_online_conv_stack_4_conv2d_13_kernel:��R
Cconv2d_13_biasadd_readvariableop_online_conv_stack_4_conv2d_13_bias:	�`
Dconv2d_14_conv2d_readvariableop_online_conv_stack_4_conv2d_14_kernel:��R
Cconv2d_14_biasadd_readvariableop_online_conv_stack_4_conv2d_14_bias:	�
identity�� conv2d_12/BiasAdd/ReadVariableOp�conv2d_12/Conv2D/ReadVariableOp� conv2d_13/BiasAdd/ReadVariableOp�conv2d_13/Conv2D/ReadVariableOp� conv2d_14/BiasAdd/ReadVariableOp�conv2d_14/Conv2D/ReadVariableOp�
conv2d_12/Conv2D/ReadVariableOpReadVariableOpDconv2d_12_conv2d_readvariableop_online_conv_stack_4_conv2d_12_kernel*(
_output_shapes
:��*
dtype02!
conv2d_12/Conv2D/ReadVariableOp�
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_12/Conv2D�
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOpCconv2d_12_biasadd_readvariableop_online_conv_stack_4_conv2d_12_bias*
_output_shapes	
:�*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp�
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_12/BiasAdd�
leaky_re_lu_10/LeakyRelu	LeakyReluconv2d_12/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_10/LeakyRelu�
conv2d_13/Conv2D/ReadVariableOpReadVariableOpDconv2d_13_conv2d_readvariableop_online_conv_stack_4_conv2d_13_kernel*(
_output_shapes
:��*
dtype02!
conv2d_13/Conv2D/ReadVariableOp�
conv2d_13/Conv2DConv2D&leaky_re_lu_10/LeakyRelu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_13/Conv2D�
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOpCconv2d_13_biasadd_readvariableop_online_conv_stack_4_conv2d_13_bias*
_output_shapes	
:�*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp�
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_13/BiasAdd�
max_pooling2d_3/MaxPoolMaxPoolconv2d_13/BiasAdd:output:0*0
_output_shapes
:����������*
ksize
*
paddingSAME*
strides
2
max_pooling2d_3/MaxPool�
conv2d_14/Conv2D/ReadVariableOpReadVariableOpDconv2d_14_conv2d_readvariableop_online_conv_stack_4_conv2d_14_kernel*(
_output_shapes
:��*
dtype02!
conv2d_14/Conv2D/ReadVariableOp�
conv2d_14/Conv2DConv2Dinputs'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_14/Conv2D�
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOpCconv2d_14_biasadd_readvariableop_online_conv_stack_4_conv2d_14_bias*
_output_shapes	
:�*
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp�
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_14/BiasAdd�
	add_4/addAddV2 max_pooling2d_3/MaxPool:output:0conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
	add_4/add�
leaky_re_lu_11/LeakyRelu	LeakyReluadd_4/add:z:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_11/LeakyRelu�
IdentityIdentity&leaky_re_lu_11/LeakyRelu:activations:0!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:����������: : : : : : 2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_2341

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
@__inference_conv2d_layer_call_and_return_conditional_losses_3085

inputsD
*conv2d_readvariableop_online_conv2d_kernel:07
)biasadd_readvariableop_online_conv2d_bias:0
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOp*conv2d_readvariableop_online_conv2d_kernel*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT0*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOp)biasadd_readvariableop_online_conv2d_bias*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������TT02	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������TT02

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������TT: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������TT
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2358

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�%
�
F__inference_conv_stack_2_layer_call_and_return_conditional_losses_3220

inputs]
Bconv2d_7_conv2d_readvariableop_online_conv_stack_2_conv2d_7_kernel:H�P
Aconv2d_7_biasadd_readvariableop_online_conv_stack_2_conv2d_7_bias:	�^
Bconv2d_8_conv2d_readvariableop_online_conv_stack_2_conv2d_8_kernel:��P
Aconv2d_8_biasadd_readvariableop_online_conv_stack_2_conv2d_8_bias:	�]
Bconv2d_9_conv2d_readvariableop_online_conv_stack_2_conv2d_9_kernel:H�P
Aconv2d_9_biasadd_readvariableop_online_conv_stack_2_conv2d_9_bias:	�
identity��conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�
conv2d_7/Conv2D/ReadVariableOpReadVariableOpBconv2d_7_conv2d_readvariableop_online_conv_stack_2_conv2d_7_kernel*'
_output_shapes
:H�*
dtype02 
conv2d_7/Conv2D/ReadVariableOp�
conv2d_7/Conv2DConv2Dinputs&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_7/Conv2D�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOpAconv2d_7_biasadd_readvariableop_online_conv_stack_2_conv2d_7_bias*
_output_shapes	
:�*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_7/BiasAdd�
leaky_re_lu_6/LeakyRelu	LeakyReluconv2d_7/BiasAdd:output:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_6/LeakyRelu�
conv2d_8/Conv2D/ReadVariableOpReadVariableOpBconv2d_8_conv2d_readvariableop_online_conv_stack_2_conv2d_8_kernel*(
_output_shapes
:��*
dtype02 
conv2d_8/Conv2D/ReadVariableOp�
conv2d_8/Conv2DConv2D%leaky_re_lu_6/LeakyRelu:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_8/Conv2D�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOpAconv2d_8_biasadd_readvariableop_online_conv_stack_2_conv2d_8_bias*
_output_shapes	
:�*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_8/BiasAdd�
max_pooling2d_2/MaxPoolMaxPoolconv2d_8/BiasAdd:output:0*0
_output_shapes
:����������*
ksize
*
paddingSAME*
strides
2
max_pooling2d_2/MaxPool�
conv2d_9/Conv2D/ReadVariableOpReadVariableOpBconv2d_9_conv2d_readvariableop_online_conv_stack_2_conv2d_9_kernel*'
_output_shapes
:H�*
dtype02 
conv2d_9/Conv2D/ReadVariableOp�
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_9/Conv2D�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOpAconv2d_9_biasadd_readvariableop_online_conv_stack_2_conv2d_9_bias*
_output_shapes	
:�*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_9/BiasAdd�
	add_2/addAddV2 max_pooling2d_2/MaxPool:output:0conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
	add_2/add�
leaky_re_lu_7/LeakyRelu	LeakyReluadd_2/add:z:0*0
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_7/LeakyRelu�
IdentityIdentity%leaky_re_lu_7/LeakyRelu:activations:0 ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������H: : : : : : 2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
�
F__inference_conv_stack_1_layer_call_and_return_conditional_losses_3185

inputs\
Bconv2d_5_conv2d_readvariableop_online_conv_stack_1_conv2d_5_kernel:HHO
Aconv2d_5_biasadd_readvariableop_online_conv_stack_1_conv2d_5_bias:H\
Bconv2d_6_conv2d_readvariableop_online_conv_stack_1_conv2d_6_kernel:HHO
Aconv2d_6_biasadd_readvariableop_online_conv_stack_1_conv2d_6_bias:H
identity��conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�
conv2d_5/Conv2D/ReadVariableOpReadVariableOpBconv2d_5_conv2d_readvariableop_online_conv_stack_1_conv2d_5_kernel*&
_output_shapes
:HH*
dtype02 
conv2d_5/Conv2D/ReadVariableOp�
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
2
conv2d_5/Conv2D�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOpAconv2d_5_biasadd_readvariableop_online_conv_stack_1_conv2d_5_bias*
_output_shapes
:H*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H2
conv2d_5/BiasAdd�
leaky_re_lu_4/LeakyRelu	LeakyReluconv2d_5/BiasAdd:output:0*/
_output_shapes
:���������H*
alpha%���>2
leaky_re_lu_4/LeakyRelu�
conv2d_6/Conv2D/ReadVariableOpReadVariableOpBconv2d_6_conv2d_readvariableop_online_conv_stack_1_conv2d_6_kernel*&
_output_shapes
:HH*
dtype02 
conv2d_6/Conv2D/ReadVariableOp�
conv2d_6/Conv2DConv2D%leaky_re_lu_4/LeakyRelu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H*
paddingSAME*
strides
2
conv2d_6/Conv2D�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOpAconv2d_6_biasadd_readvariableop_online_conv_stack_1_conv2d_6_bias*
_output_shapes
:H*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������H2
conv2d_6/BiasAdd|
	add_1/addAddV2conv2d_6/BiasAdd:output:0inputs*
T0*/
_output_shapes
:���������H2
	add_1/add�
leaky_re_lu_5/LeakyRelu	LeakyReluadd_1/add:z:0*/
_output_shapes
:���������H*
alpha%���>2
leaky_re_lu_5/LeakyRelu�
IdentityIdentity%leaky_re_lu_5/LeakyRelu:activations:0 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������H2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������H: : : : 2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������H
 
_user_specified_nameinputs
�
H
,__inference_leaky_re_lu_1_layer_call_fn_3129

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������))0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_24622
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������))02

Identity"
identityIdentity:output:0*.
_input_shapes
:���������))0:W S
/
_output_shapes
:���������))0
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2367

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������TT<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
main_layers
regularization_losses
	variables
trainable_variables
	keras_api

signatures
�_default_save_signature
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_model�{"name": "Online", "trainable": true, "expects_training_arg": false, "dtype": "uint8", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "SArchDQNNetwork", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 84, 84, 4]}, "uint8", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "SArchDQNNetwork"}}
~
0
1
	2

3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
)21
*22
+23
,24
-25
.26
/27
028
129
230
331
432
533
634
735"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
)21
*22
+23
,24
-25
.26
/27
028
129
230
331
432
533
634
735"
trackable_list_wrapper
�
regularization_losses

8layers
	variables
9layer_metrics
:metrics
;non_trainable_variables
<layer_regularization_losses
trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�


kernel
bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 84, 84, 4]}}
�
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 4}
�


kernel
bias
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}, "shared_object_id": 8}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 84, 84, 48]}}
�
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 10}}
�
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 11}
�
Qmain_layers
Rskip_layers
Sjoin_layers
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "conv_stack", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ConvStack", "config": {"layer was saved without config": true}}
�
Xmain_layers
Yskip_layers
Zjoin_layers
[regularization_losses
\	variables
]trainable_variables
^	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "conv_stack_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ConvStack", "config": {"layer was saved without config": true}}
�
_main_layers
`skip_layers
ajoin_layers
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "conv_stack_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ConvStack", "config": {"layer was saved without config": true}}
�
fmain_layers
gskip_layers
hjoin_layers
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "conv_stack_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ConvStack", "config": {"layer was saved without config": true}}
�
mmain_layers
nskip_layers
ojoin_layers
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "conv_stack_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ConvStack", "config": {"layer was saved without config": true}}
�
tmain_layers
uskip_layers
vjoin_layers
wregularization_losses
x	variables
ytrainable_variables
z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "conv_stack_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ConvStack", "config": {"layer was saved without config": true}}
�
{regularization_losses
|	variables
}trainable_variables
~	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 13}}
�

6kernel
7bias
regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 18, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 248}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 248]}}
.:,02Online/conv2d/kernel
 :02Online/conv2d/bias
0:.002Online/conv2d_1/kernel
": 02Online/conv2d_1/bias
;:90H2!Online/conv_stack/conv2d_2/kernel
-:+H2Online/conv_stack/conv2d_2/bias
;:9HH2!Online/conv_stack/conv2d_3/kernel
-:+H2Online/conv_stack/conv2d_3/bias
;:90H2!Online/conv_stack/conv2d_4/kernel
-:+H2Online/conv_stack/conv2d_4/bias
=:;HH2#Online/conv_stack_1/conv2d_5/kernel
/:-H2!Online/conv_stack_1/conv2d_5/bias
=:;HH2#Online/conv_stack_1/conv2d_6/kernel
/:-H2!Online/conv_stack_1/conv2d_6/bias
>:<H�2#Online/conv_stack_2/conv2d_7/kernel
0:.�2!Online/conv_stack_2/conv2d_7/bias
?:=��2#Online/conv_stack_2/conv2d_8/kernel
0:.�2!Online/conv_stack_2/conv2d_8/bias
>:<H�2#Online/conv_stack_2/conv2d_9/kernel
0:.�2!Online/conv_stack_2/conv2d_9/bias
@:>��2$Online/conv_stack_3/conv2d_10/kernel
1:/�2"Online/conv_stack_3/conv2d_10/bias
@:>��2$Online/conv_stack_3/conv2d_11/kernel
1:/�2"Online/conv_stack_3/conv2d_11/bias
@:>��2$Online/conv_stack_4/conv2d_12/kernel
1:/�2"Online/conv_stack_4/conv2d_12/bias
@:>��2$Online/conv_stack_4/conv2d_13/kernel
1:/�2"Online/conv_stack_4/conv2d_13/bias
@:>��2$Online/conv_stack_4/conv2d_14/kernel
1:/�2"Online/conv_stack_4/conv2d_14/bias
@:>��2$Online/conv_stack_5/conv2d_15/kernel
1:/�2"Online/conv_stack_5/conv2d_15/bias
@:>��2$Online/conv_stack_5/conv2d_16/kernel
1:/�2"Online/conv_stack_5/conv2d_16/bias
&:$	�2Online/dense/kernel
:2Online/dense/bias
~
0
1
	2

3
4
5
6
7
8
9
10
11
12"
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
=regularization_losses
�layers
>	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
?trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Aregularization_losses
�layers
B	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
Ctrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Eregularization_losses
�layers
F	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
Gtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Iregularization_losses
�layers
J	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
Ktrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Mregularization_losses
�layers
N	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
Otrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
@
�0
�1
�2
�3"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
�
Tregularization_losses
�layers
U	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
Vtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
 2
!3"
trackable_list_wrapper
<
0
1
 2
!3"
trackable_list_wrapper
�
[regularization_losses
�layers
\	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
]trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
@
�0
�1
�2
�3"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
J
"0
#1
$2
%3
&4
'5"
trackable_list_wrapper
J
"0
#1
$2
%3
&4
'5"
trackable_list_wrapper
�
bregularization_losses
�layers
c	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
dtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
(0
)1
*2
+3"
trackable_list_wrapper
<
(0
)1
*2
+3"
trackable_list_wrapper
�
iregularization_losses
�layers
j	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
ktrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
@
�0
�1
�2
�3"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
J
,0
-1
.2
/3
04
15"
trackable_list_wrapper
J
,0
-1
.2
/3
04
15"
trackable_list_wrapper
�
pregularization_losses
�layers
q	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
rtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
20
31
42
53"
trackable_list_wrapper
<
20
31
42
53"
trackable_list_wrapper
�
wregularization_losses
�layers
x	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
ytrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
{regularization_losses
�layers
|	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
}trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
�
regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
�


kernel
bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 41, 41, 48]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 22}
�


kernel
bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 72}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 41, 41, 72]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 28}}
�


kernel
bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 41, 41, 48]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "shared_object_id": 33, "build_input_shape": [{"class_name": "TensorShape", "items": [1, 21, 21, 72]}, {"class_name": "TensorShape", "items": [1, 21, 21, 72]}]}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 34}
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�


kernel
bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 72}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 21, 21, 72]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 39}
�


 kernel
!bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 40}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 42, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 72}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 21, 21, 72]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "shared_object_id": 44, "build_input_shape": [{"class_name": "TensorShape", "items": [1, 21, 21, 72]}, {"class_name": "TensorShape", "items": [1, 21, 21, 72]}]}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 45}
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�


"kernel
#bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 46}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 47}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 48, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 72}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 21, 21, 72]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 50}
�


$kernel
%bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 51}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 52}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 53, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 144}}, "shared_object_id": 54}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 21, 21, 144]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 55, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 56}}
�


&kernel
'bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 57}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 58}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 59, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 72}}, "shared_object_id": 60}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 21, 21, 72]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "shared_object_id": 61, "build_input_shape": [{"class_name": "TensorShape", "items": [1, 11, 11, 144]}, {"class_name": "TensorShape", "items": [1, 11, 11, 144]}]}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 62}
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�


(kernel
)bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 63}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 64}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 65, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 144}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 11, 144]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_8", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 67}
�


*kernel
+bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 144, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 68}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 69}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 70, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 144}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 11, 144]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "shared_object_id": 72, "build_input_shape": [{"class_name": "TensorShape", "items": [1, 11, 11, 144]}, {"class_name": "TensorShape", "items": [1, 11, 11, 144]}]}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_re_lu_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_9", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 73}
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�


,kernel
-bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 248, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 74}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 75}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 76, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 144}}, "shared_object_id": 77}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 11, 144]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_re_lu_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_10", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 78}
�


.kernel
/bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 248, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 79}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 80}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 81, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 248}}, "shared_object_id": 82}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 11, 248]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 83, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 84}}
�


0kernel
1bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 248, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 85}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 86}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 87, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 144}}, "shared_object_id": 88}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 11, 144]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "add_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_4", "trainable": true, "dtype": "float32"}, "shared_object_id": 89, "build_input_shape": [{"class_name": "TensorShape", "items": [1, 6, 6, 248]}, {"class_name": "TensorShape", "items": [1, 6, 6, 248]}]}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_re_lu_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_11", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 90}
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�


2kernel
3bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 248, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 91}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 92}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 93, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 248}}, "shared_object_id": 94}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 6, 6, 248]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_re_lu_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_12", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 95}
�


4kernel
5bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "dtype": "float32", "filters": 248, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 96}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}, "shared_object_id": 97}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 98, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 248}}, "shared_object_id": 99}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 6, 6, 248]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "add_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_5", "trainable": true, "dtype": "float32"}, "shared_object_id": 100, "build_input_shape": [{"class_name": "TensorShape", "items": [1, 6, 6, 248]}, {"class_name": "TensorShape", "items": [1, 6, 6, 248]}]}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_re_lu_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_13", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "shared_object_id": 101}
H
�0
�1
�2
�3
�4"
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�regularization_losses
�layers
�	variables
�layer_metrics
�metrics
�non_trainable_variables
 �layer_regularization_losses
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
�2�
__inference__wrapped_model_2327�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������TT
�2�
@__inference_Online_layer_call_and_return_conditional_losses_2657�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������TT
�2�
%__inference_Online_layer_call_fn_2755�
���
FullArgSpec
args�
jself
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������TT
�B�
"__inference_signature_wrapper_3075input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_conv2d_layer_call_and_return_conditional_losses_3085�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_conv2d_layer_call_fn_3092�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_leaky_re_lu_layer_call_and_return_conditional_losses_3097�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_leaky_re_lu_layer_call_fn_3102�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_3112�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv2d_1_layer_call_fn_3119�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_2333�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
,__inference_max_pooling2d_layer_call_fn_2344�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
G__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_3124�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_leaky_re_lu_1_layer_call_fn_3129�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv_stack_layer_call_and_return_conditional_losses_3155�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv_stack_layer_call_fn_3166�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_conv_stack_1_layer_call_and_return_conditional_losses_3185�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv_stack_1_layer_call_fn_3194�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_conv_stack_2_layer_call_and_return_conditional_losses_3220�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv_stack_2_layer_call_fn_3231�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_conv_stack_3_layer_call_and_return_conditional_losses_3250�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv_stack_3_layer_call_fn_3259�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_conv_stack_4_layer_call_and_return_conditional_losses_3285�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv_stack_4_layer_call_fn_3296�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_conv_stack_5_layer_call_and_return_conditional_losses_3315�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_conv_stack_5_layer_call_fn_3324�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_2402�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
7__inference_global_average_pooling2d_layer_call_fn_2414�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
?__inference_dense_layer_call_and_return_conditional_losses_3334�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_dense_layer_call_fn_3341�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2350�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
.__inference_max_pooling2d_1_layer_call_fn_2361�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2367�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
.__inference_max_pooling2d_2_layer_call_fn_2378�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2384�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
.__inference_max_pooling2d_3_layer_call_fn_2395�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
@__inference_Online_layer_call_and_return_conditional_losses_2657�$ !"#$%&'()*+,-./012345678�5
.�+
)�&
input_1���������TT
� "L�I
B�?
dqn_network0
q_values$�!

0/q_values���������
� �
%__inference_Online_layer_call_fn_2755�$ !"#$%&'()*+,-./012345678�5
.�+
)�&
input_1���������TT
� "@�=
dqn_network.
q_values"�
q_values����������
__inference__wrapped_model_2327�$ !"#$%&'()*+,-./012345678�5
.�+
)�&
input_1���������TT
� "3�0
.
output_1"�
output_1����������
B__inference_conv2d_1_layer_call_and_return_conditional_losses_3112l7�4
-�*
(�%
inputs���������TT0
� "-�*
#� 
0���������TT0
� �
'__inference_conv2d_1_layer_call_fn_3119_7�4
-�*
(�%
inputs���������TT0
� " ����������TT0�
@__inference_conv2d_layer_call_and_return_conditional_losses_3085l7�4
-�*
(�%
inputs���������TT
� "-�*
#� 
0���������TT0
� �
%__inference_conv2d_layer_call_fn_3092_7�4
-�*
(�%
inputs���������TT
� " ����������TT0�
F__inference_conv_stack_1_layer_call_and_return_conditional_losses_3185n !7�4
-�*
(�%
inputs���������H
� "-�*
#� 
0���������H
� �
+__inference_conv_stack_1_layer_call_fn_3194a !7�4
-�*
(�%
inputs���������H
� " ����������H�
F__inference_conv_stack_2_layer_call_and_return_conditional_losses_3220q"#$%&'7�4
-�*
(�%
inputs���������H
� ".�+
$�!
0����������
� �
+__inference_conv_stack_2_layer_call_fn_3231d"#$%&'7�4
-�*
(�%
inputs���������H
� "!������������
F__inference_conv_stack_3_layer_call_and_return_conditional_losses_3250p()*+8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
+__inference_conv_stack_3_layer_call_fn_3259c()*+8�5
.�+
)�&
inputs����������
� "!������������
F__inference_conv_stack_4_layer_call_and_return_conditional_losses_3285r,-./018�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
+__inference_conv_stack_4_layer_call_fn_3296e,-./018�5
.�+
)�&
inputs����������
� "!������������
F__inference_conv_stack_5_layer_call_and_return_conditional_losses_3315p23458�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
+__inference_conv_stack_5_layer_call_fn_3324c23458�5
.�+
)�&
inputs����������
� "!������������
D__inference_conv_stack_layer_call_and_return_conditional_losses_3155p7�4
-�*
(�%
inputs���������))0
� "-�*
#� 
0���������H
� �
)__inference_conv_stack_layer_call_fn_3166c7�4
-�*
(�%
inputs���������))0
� " ����������H�
?__inference_dense_layer_call_and_return_conditional_losses_3334]670�-
&�#
!�
inputs����������
� "%�"
�
0���������
� x
$__inference_dense_layer_call_fn_3341P670�-
&�#
!�
inputs����������
� "�����������
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_2402�R�O
H�E
C�@
inputs4������������������������������������
� ".�+
$�!
0������������������
� �
7__inference_global_average_pooling2d_layer_call_fn_2414wR�O
H�E
C�@
inputs4������������������������������������
� "!��������������������
G__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_3124h7�4
-�*
(�%
inputs���������))0
� "-�*
#� 
0���������))0
� �
,__inference_leaky_re_lu_1_layer_call_fn_3129[7�4
-�*
(�%
inputs���������))0
� " ����������))0�
E__inference_leaky_re_lu_layer_call_and_return_conditional_losses_3097h7�4
-�*
(�%
inputs���������TT0
� "-�*
#� 
0���������TT0
� �
*__inference_leaky_re_lu_layer_call_fn_3102[7�4
-�*
(�%
inputs���������TT0
� " ����������TT0�
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_2350�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
.__inference_max_pooling2d_1_layer_call_fn_2361�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_2367�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
.__inference_max_pooling2d_2_layer_call_fn_2378�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_2384�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
.__inference_max_pooling2d_3_layer_call_fn_2395�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_2333�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
,__inference_max_pooling2d_layer_call_fn_2344�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
"__inference_signature_wrapper_3075�$ !"#$%&'()*+,-./01234567C�@
� 
9�6
4
input_1)�&
input_1���������TT"3�0
.
output_1"�
output_1���������