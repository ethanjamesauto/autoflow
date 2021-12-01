# AutoFlow
A machine learning framework for learning/testing machine learning concepts

# Features
This framework takes advantage of the relatively high-level data structures included in the C++11 standard, such as ```std::shared_ptr```.
This means that there is no need to worry about the "rule of 3/5/0", as the pointers are automatically tracked using reference counting
with minimal overhead.

The Tensor class is mutable for efficiency reasons, however for every mutable method there exists an immutable method that is marked as ```const```.

# Resources Used
https://www.tensorflow.org/api_docs \
https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/readings/gradient-notes.pdf \
Deep Learning with Python, Second Edition - Francois Chollet
