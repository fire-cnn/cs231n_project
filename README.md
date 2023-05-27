# A table is worth a thousand words: multi-modal learning in image classification with tabular data

This repo contains the code for the CS231N final project. We develop a image
classification pipeline that uses both text and image embeddings, and a 
contrastive loss to bring image and text into the same embedding space and 
augment visual data for classification. 

## Tabular-2-prompt strategies

Following (TabLLM)[1], we use two prompting strategies: 

1. Bank list: 

```
 - Year Built: 1962.0
 - Aspect: N
 - Elevation: 2223
 - Slope: 12.04
 - Temperature: 297.73
 - Fuel Moisture: 6.0
 - Vapor Pressure Deficit: 1.56
 - Evapotranspiration: 7.31
 - Precipitation: 0.0
 - Humidity: 44.48
 - Specific Humidity: 0.0
 - Shortwave Flux: 332.73
 - Wind Direction: 159.71
 - Wind Speed: 2.3
 - Age: 58.0
 - Risk to structure: 0.09
 - Fire Name: Creek 
```

2. Custom template: 

```
This house is 58.0 years old. It is located 2223 meters above sea level with
a slope of 12.04. Temperature is 297.73 degrees. Relative humidity is 44.48.
Wind speed is 2.3. The vapor pressure deficit is 1.56 and the fuel moisture was
6.0. The risk to structure is 0.09. The fire name is Creek
```

Both strategies have a common final prompt passed during fine-tuning of the LLM:
`Does this house will be destroyed? Yes or No? Answer:`. This will be the final 
prompt for classification. 

[1]: https://arxiv.org/abs/2210.10723
