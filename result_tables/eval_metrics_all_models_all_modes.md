# Evaluation Metrics Summary

## qwen2.5-1.5b-instruct

### 2wikimultihopqa

| data_type | mode | em | f1 | prec | recall |
|---|---:|---:|---:|---:|---:|
| inference | vanilla | 0.0067 | 0.1704 | 0.1884 | 0.1811 |
| inference | icl | 0.0033 | 0.1187 | 0.1108 | 0.1711 |
| inference | prag | 0.0200 | 0.2007 | 0.2251 | 0.2104 |
| inference | combine | 0.0100 | 0.1638 | 0.1647 | 0.2070 |
| comparison | vanilla | 0.4133 | 0.4574 | 0.4488 | 0.5060 |
| comparison | icl | 0.2600 | 0.3875 | 0.3498 | 0.5352 |
| comparison | prag | 0.3867 | 0.4384 | 0.4269 | 0.4891 |
| comparison | combine | 0.2833 | 0.3986 | 0.3666 | 0.5162 |
| bridge_comparison | vanilla | 0.3700 | 0.3906 | 0.3952 | 0.3951 |
| bridge_comparison | icl | 0.3533 | 0.3884 | 0.3908 | 0.4014 |
| bridge_comparison | prag | 0.4233 | 0.4487 | 0.4555 | 0.4491 |
| bridge_comparison | combine | 0.4000 | 0.4353 | 0.4375 | 0.4477 |
| compositional | vanilla | 0.0433 | 0.0727 | 0.0795 | 0.0763 |
| compositional | icl | 0.0300 | 0.0568 | 0.0609 | 0.0725 |
| compositional | prag | 0.0600 | 0.1042 | 0.1153 | 0.1015 |
| compositional | combine | 0.0500 | 0.0967 | 0.1021 | 0.1200 |
### hotpotqa

| data_type | mode | em | f1 | prec | recall |
|---|---:|---:|---:|---:|---:|
| comparison | vanilla | 0.3067 | 0.3946 | 0.3932 | 0.4901 |
| comparison | icl | 0.2467 | 0.3713 | 0.3497 | 0.5027 |
| comparison | prag | 0.3667 | 0.4508 | 0.4639 | 0.4648 |
| comparison | combine | 0.3433 | 0.4332 | 0.4419 | 0.4722 |
| bridge | vanilla | 0.0467 | 0.1218 | 0.1244 | 0.1578 |
| bridge | icl | 0.0867 | 0.1619 | 0.1502 | 0.2479 |
| bridge | prag | 0.0900 | 0.1664 | 0.1741 | 0.1774 |
| bridge | combine | 0.1433 | 0.2240 | 0.2229 | 0.2573 |
### complexwebquestions

| data_type | mode | em | f1 | prec | recall |
|---|---:|---:|---:|---:|---:|
| total | vanilla | 0.1800 | 0.2647 | 0.2665 | 0.2853 |
| total | icl | 0.1833 | 0.2823 | 0.3001 | 0.3006 |
| total | prag | 0.2133 | 0.2996 | 0.3073 | 0.3156 |
| total | combine | 0.2300 | 0.3342 | 0.3581 | 0.3438 |
### popqa

| data_type | mode | em | f1 | prec | recall |
|---|---:|---:|---:|---:|---:|
| total | vanilla | 0.0067 | 0.0298 | 0.0213 | 0.0733 |
| total | icl | 0.0067 | 0.0994 | 0.0646 | 0.3050 |
| total | prag | 0.1700 | 0.2133 | 0.1981 | 0.2783 |
| total | combine | 0.1200 | 0.2430 | 0.2042 | 0.4050 |
## llama3.2-1b-instruct

### 2wikimultihopqa

| data_type | mode | em | f1 | prec | recall |
|---|---:|---:|---:|---:|---:|
| inference | vanilla | 0.0067 | 0.1694 | 0.1873 | 0.1743 |
| inference | icl | 0.0300 | 0.2263 | 0.2514 | 0.2390 |
| inference | prag | 0.0033 | 0.1913 | 0.2209 | 0.1838 |
| inference | combine | 0.0233 | 0.2351 | 0.2638 | 0.2379 |
| comparison | vanilla | 0.4200 | 0.4346 | 0.4374 | 0.4341 |
| comparison | icl | 0.4100 | 0.4298 | 0.4298 | 0.4385 |
| comparison | prag | 0.4833 | 0.5021 | 0.5041 | 0.5024 |
| comparison | combine | 0.4933 | 0.5159 | 0.5166 | 0.5229 |
| bridge_comparison | vanilla | 0.2300 | 0.2417 | 0.2468 | 0.2400 |
| bridge_comparison | icl | 0.2833 | 0.3032 | 0.3066 | 0.3069 |
| bridge_comparison | prag | 0.4033 | 0.4190 | 0.4264 | 0.4162 |
| bridge_comparison | combine | 0.4267 | 0.4466 | 0.4507 | 0.4522 |
| compositional | vanilla | 0.0333 | 0.0697 | 0.0777 | 0.0728 |
| compositional | icl | 0.0567 | 0.1064 | 0.1171 | 0.1183 |
| compositional | prag | 0.0700 | 0.1037 | 0.1133 | 0.1018 |
| compositional | combine | 0.0767 | 0.1165 | 0.1279 | 0.1169 |
### hotpotqa

| data_type | mode | em | f1 | prec | recall |
|---|---:|---:|---:|---:|---:|
| comparison | vanilla | 0.3333 | 0.3977 | 0.4149 | 0.4142 |
| comparison | icl | 0.3267 | 0.4083 | 0.4158 | 0.4570 |
| comparison | prag | 0.3867 | 0.4466 | 0.4591 | 0.4473 |
| comparison | combine | 0.3767 | 0.4516 | 0.4684 | 0.4625 |
| bridge | vanilla | 0.0833 | 0.1275 | 0.1232 | 0.1540 |
| bridge | icl | 0.1200 | 0.2110 | 0.2116 | 0.2649 |
| bridge | prag | 0.1133 | 0.1658 | 0.1769 | 0.1658 |
| bridge | combine | 0.1833 | 0.2584 | 0.2720 | 0.2762 |
### complexwebquestions

| data_type | mode | em | f1 | prec | recall |
|---|---:|---:|---:|---:|---:|
| total | vanilla | 0.2300 | 0.3403 | 0.3422 | 0.3659 |
| total | icl | 0.2500 | 0.3726 | 0.3833 | 0.3979 |
| total | prag | 0.2467 | 0.3456 | 0.3502 | 0.3689 |
| total | combine | 0.2800 | 0.3916 | 0.4042 | 0.4127 |
### popqa

| data_type | mode | em | f1 | prec | recall |
|---|---:|---:|---:|---:|---:|
| total | vanilla | 0.0067 | 0.0263 | 0.0207 | 0.0611 |
| total | icl | 0.0967 | 0.1867 | 0.1557 | 0.3533 |
| total | prag | 0.1500 | 0.1754 | 0.1686 | 0.2111 |
| total | combine | 0.2067 | 0.3113 | 0.2808 | 0.4450 |
## llama3-8b-instruct

### 2wikimultihopqa

| data_type | mode | em | f1 | prec | recall |
|---|---:|---:|---:|---:|---:|
| inference | vanilla | 0.0467 | 0.2400 | 0.2606 | 0.2503 |
| inference | icl | 0.0667 | 0.1833 | 0.1908 | 0.2173 |
| inference | prag | 0.0600 | 0.2472 | 0.2874 | 0.2391 |
| inference | combine | 0.1000 | 0.3265 | 0.3717 | 0.3238 |
| comparison | vanilla | 0.4900 | 0.5068 | 0.5120 | 0.5097 |
| comparison | icl | 0.5433 | 0.5876 | 0.5871 | 0.6068 |
| comparison | prag | 0.6000 | 0.6162 | 0.6197 | 0.6141 |
| comparison | combine | 0.6000 | 0.6326 | 0.6351 | 0.6382 |
| bridge_comparison | vanilla | 0.5067 | 0.5253 | 0.5305 | 0.5279 |
| bridge_comparison | icl | 0.4433 | 0.4794 | 0.4853 | 0.4888 |
| bridge_comparison | prag | 0.6033 | 0.6223 | 0.6299 | 0.6209 |
| bridge_comparison | combine | 0.5500 | 0.5775 | 0.5808 | 0.5813 |
| compositional | vanilla | 0.1000 | 0.1405 | 0.1478 | 0.1428 |
| compositional | icl | 0.0700 | 0.0991 | 0.1115 | 0.0979 |
| compositional | prag | 0.1300 | 0.1919 | 0.2107 | 0.1885 |
| compositional | combine | 0.1600 | 0.2006 | 0.2154 | 0.2036 |
### hotpotqa

| data_type | mode | em | f1 | prec | recall |
|---|---:|---:|---:|---:|---:|
| comparison | vanilla | 0.3300 | 0.4524 | 0.4566 | 0.5215 |
| comparison | icl | 0.2233 | 0.3493 | 0.3424 | 0.4374 |
| comparison | prag | 0.5467 | 0.6413 | 0.6683 | 0.6394 |
| comparison | combine | 0.5800 | 0.6838 | 0.7161 | 0.6843 |
| bridge | vanilla | 0.0900 | 0.1560 | 0.1435 | 0.2220 |
| bridge | icl | 0.1167 | 0.1823 | 0.1781 | 0.2450 |
| bridge | prag | 0.2233 | 0.3360 | 0.3564 | 0.3371 |
| bridge | combine | 0.2667 | 0.3835 | 0.3985 | 0.3975 |
### complexwebquestions

| data_type | mode | em | f1 | prec | recall |
|---|---:|---:|---:|---:|---:|
| total | vanilla | 0.3067 | 0.4292 | 0.4445 | 0.4677 |
| total | icl | 0.2533 | 0.3545 | 0.3671 | 0.3817 |
| total | prag | 0.3133 | 0.4177 | 0.4422 | 0.4320 |
| total | combine | 0.2700 | 0.3602 | 0.3761 | 0.3807 |
### popqa

| data_type | mode | em | f1 | prec | recall |
|---|---:|---:|---:|---:|---:|
| total | vanilla | 0.0500 | 0.0847 | 0.0722 | 0.1511 |
| total | icl | 0.0600 | 0.1646 | 0.1336 | 0.3000 |
| total | prag | 0.1500 | 0.2326 | 0.2061 | 0.3650 |
| total | combine | 0.1533 | 0.2827 | 0.2433 | 0.4517 |

