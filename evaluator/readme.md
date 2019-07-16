# StreamLAB

StreamLAB is a monitoring framework for reactive systems.


## Command Line Usage

### Specification Analysis

```
streamlab analyze [SPEC]
```

checks whether the given specification is valid

### Monitoring

```
streamlab monitor [SPEC] --offline --csv-in [TRACE] --verbosity progress
```

For example, given the specification

```
input a: Int64
input b: Int64

output x := a + b
trigger x > 2
```

in file `example.spec` and the CSV

```
a,b,time
0,1,0.1
2,3,0.2
4,5,0.3
```

in file `example.csv` we get

```
streamlab monitor example.spec --offline --csv-in example.csv 
Trigger: x > 2
Trigger: x > 2
```


See all available options with `streamlab --help`