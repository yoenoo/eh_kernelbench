for each output location (h, w) in the output map:
       out_val = 0
       for each input channel:
           for each kernel row and column:
               if within input boundaries:
                   out_val += input[...] * kernel[...]
       out[...] = out_val + bias