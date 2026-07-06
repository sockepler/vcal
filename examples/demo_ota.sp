* Demo: 5-transistor OTA — 通用示例网表(展示配置格式用)
* Generic 5T OTA — replace n_example/p_example with YOUR PDK device
* models (see pdks/example.yaml). Not meant to run as-is.
.subckt demo_ota vdd vss vinn vinp vout vbias
  Mtail net1 vbias vss  vss  n_example w=10u l=1u   nf=1
  Min1  outi vinn net1  vss  n_example w=20u l=0.5u nf=1
  Min2  vout vinp net1  vss  n_example w=20u l=0.5u nf=1
  Mld1  outi outi vdd   vdd  p_example w=10u l=0.5u nf=1
  Mld2  vout outi vdd   vdd  p_example w=10u l=0.5u nf=1
.ends demo_ota
