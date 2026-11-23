# **Complete Lesson: Artificial Neural Networks (ANN) – Customer Churn Prediction**

We are going to cover the entire 27-page PDF, from the introduction to the final business interpretation.

I'll teach it as a beginner-friendly classroom lesson. We won't just memorize definitions. We'll understand what each formula means, why we use it, and how to calculate it.

**Our learning objectives:**

By the end, you should be able to:

1. Explain how Artificial Neural Networks work.
2. Identify the different parts of an artificial neuron.
3. Explain forward propagation and backpropagation.
4. Calculate weighted sums and sigmoid probabilities.
5. Calculate prediction errors using a loss function.
6. Explain the chain rule and calculate gradients.
7. Perform gradient descent and update weights.
8. Interpret the results in a business environment.

# **PART 1: INTRODUCTION TO ARTIFICIAL NEURAL NETWORKS**

## **1. What is an Artificial Neural Network?**

An Artificial Neural Network (ANN) is a computing system inspired by the human brain.

The human brain contains biological neurons that receive information, process it and send signals to other neurons.

An ANN imitates this idea using artificial neurons connected together.

**Example:**

Imagine an online shopping company wants to know whether a customer is likely to stop buying from its website.

The company has historical information about:

- Customer purchases.
- Amount of money spent.
- Number of website visits.
- Customer activity.
- Previous shopping behaviour.

An ANN studies these patterns and learns to predict which customers may leave.

This is called customer churn prediction.

In your PDF, the worked mathematical example uses cart abandonment: predicting whether a shopper will leave without completing a purchase.

## **2. The three layers of an ANN**

[What are AI agents? Technologies, architecture & capabilities](https://images.openai.com/static-rsc-4/6SiaYTxn6xzO5v36XslEV7LvBhQMhZeHIG8dDbKvZSsUv6RLTaAtW6BsWVYXAbkszVQDj1AWhskzWyAj77KngOgy2U_uRUkdopQRapjUqMbbJ-QIXiFPM7seDNKlB2XBFbg97tXmoE9q6NLTtCGFn3knGhnCtklDLh1SyWEVEQE?purpose=inline)

[manageengine.com](https://www.manageengine.com/academy/what-are-ai-agents.html?utm_source=chatgpt.com)

An ANN has three main types of layers.

**A. Input layer**

This is where the network receives information.

For example:

- Age = 25
- Purchases = 4
- Spending = 200

The information entering the network is called input data.

**B. Hidden layer**

This is where processing takes place.

The network examines the information and learns relationships between different inputs.

For example, it might learn that customers who purchase less frequently have a higher likelihood of leaving.

There can be one or several hidden layers.

**C. Output layer**

This produces the final prediction.

For example:

- 0 = Customer does not abandon the cart.
- 1 = Customer abandons the cart.

A probability-based model can also produce a result such as 0.85, representing an estimated 85% probability.

## **3. Why do we use ANNs?**

Your PDF identifies three major advantages:

1. They learn complex patterns that are difficult to program manually.
2. They adapt and improve as they receive more training data.
3. They can handle complicated and noisy information.

Examples include facial recognition, speech recognition and language translation.

## **4. The two main training processes**

This is extremely important.

**Forward propagation**

Prediction process

[Introduction to Neural Networks for Advanced Deep Learning(Part 2).](https://images.openai.com/static-rsc-4/1e6agcxTlNpSxuRQNlM58xDkMReBVKlBesM7_ueW9TqM-t7przWLzAvA1fMtDR5KxyujseBk8VN91GRo2EijW0SWe4VeoDtkw14Kx9kSPOSQwhFW1Ycft4tNgK2vSUpdpq5XpLhHGZEn-jq_ozqX-KpMX7h_eqoEMUAJv3zi3ng?purpose=inline)

[blog.neurotech.africa](https://blog.neurotech.africa/introduction-to-neural-networks-for-advanced-deep-learning/?utm_source=chatgpt.com)

Information moves from the input layer through the hidden layers to the output layer.

The network makes a prediction.

**Backward propagation**

Learning process

[Fundamentals of Deep Learning. Deep Learning and core concepts —… | by Pavitra Dumpala | Medium](https://images.openai.com/static-rsc-4/XzhiBD-p7bBJ7gM5zXyLW2IxybFdaptEakdMfVXubfKSH9wBMuVZ-lFN0_7ohTrafJJpBK0pqN-FVY_3sSghiZnvc-Z3SgGXrcDf28hgumaNuw-0yKuoUjYUo0gpBVxLLeFrAM5MU8Jm4D9FLqWluqeBV1zQak7i96YkRV9FER8?purpose=inline)

[medium.com](https://medium.com/%40pavitrarao2274/fundamentals-of-deep-learning-e37aa27955e0?utm_source=chatgpt.com)

The network compares its prediction with the correct answer.

It calculates how its weights and bias should change to reduce the error.

Think about it this way:

- Forward propagation = Taking an examination and giving an answer.
- Backpropagation = Receiving feedback about your mistakes and learning how to improve.

These processes work together during training.

## **5. Parts of an artificial neuron**

Your PDF compares biological neurons with artificial neurons on pages 2–4.

[Artificial Neural Networks: A Complete Beginner's Guide - Wasil Zafar](https://images.openai.com/static-rsc-4/cxXwkCCjWWX_ujO2lutwJPdZIlewWRmTYXh4m6vFyZTXPLHvTLVQLWivl5GOGCeDjXMWbtLaLxV2rlwIckOlAtJiBDYLAjPt76g6i62ZJjEu4FVUYe1RtctbV9yySI6U0Qr7P1X6z2iDqqsz1gAaLXNWUnM4z34kUvi1v4m6tX4?purpose=inline)

[wasilzafar.com](https://www.wasilzafar.com/pages/series/ai-data-science/artificial-neural-networks-guide.html?utm_source=chatgpt.com)

Let's understand each part.

| Biological part | Artificial equivalent | Meaning                 |
| --------------- | --------------------- | ----------------------- |
| Dendrites       | Inputs (x)            | Receive information     |
| Synapses        | Weights (w)           | Determine importance    |
| Cell body       | Weighted sum          | Combines information    |
| Nucleus analogy | Bias (b)              | Adjusts the calculation |
| Axon hillock    | Activation function   | Determines the output   |
| Axon            | Propagation path      | Carries the signal      |
| Axon terminals  | Output (y)            | Sends the result        |

### **A. Inputs (x)**

Inputs are the numbers entering the neuron.

For example:

\\[ x_1=2,\quad x_2=-1,\quad x_3=3 \\]

### **B. Weights (w)**

Weights determine how strongly each input influences the result.

For example:

\\[ w_1=0.8,\quad w_2=0.3,\quad w_3=0.5 \\]

A larger absolute weight means a stronger influence in the calculation.

A positive weight pushes the weighted sum upward, while a negative weight pushes it downward.

### **C. Bias (b)**

Bias is an additional number added to the weighted sum.

For example:

\\[ b=0.5 \\]

It helps shift the neuron's output.

### **D. Activation function**

An activation function determines what output the neuron produces from its calculated total.

Your PDF introduces three examples:

- Step function.
- Sigmoid function.
- ReLU function.

### **E. Output (y)**

This is the final result produced by the neuron.

It can be passed to another neuron or become the network's final prediction.

## **6. Numerical example: Calculating a neuron's output**

Your PDF gives this example on pages 3–5.

We have three inputs:

| Input | Value | Weight |
| ----- | ----- | ------ |
| Alice | 2     | 0.8    |
| Bob   | -1    | 0.3    |
| Carol | 3     | 0.5    |

First, multiply each input by its weight.

\\[ (2\times0.8)=1.6 \\]

\\[ (-1\times0.3)=-0.3 \\]

\\[ (3\times0.5)=1.5 \\]

Add the results:

\\[ 1.6-0.3+1.5=2.8 \\]

Add the bias:

\\[ 2.8+0.5=3.3 \\]

The threshold is 3.0.

Since:

\\[ 3.3>3.0 \\]

The neuron fires.

\\[ \boxed{Output=1} \\]

The important lesson: inputs are multiplied by weights, combined, adjusted by bias and passed through an activation function.

# **PART 2: THE BUSINESS PROBLEM – CART ABANDONMENT**

## **7. Understanding the problem**

Your PDF introduces the business case on page 5.

Imagine an e-commerce company.

A customer visits its website, selects several products and adds them to their shopping cart.

However, the customer leaves without completing the purchase.

This is called cart abandonment.

The company wants to predict which shoppers are likely to abandon their carts.

Why?

Because it can:

- Offer incentives or discounts.
- Send personalized reminders.
- Improve the checkout experience.
- Reduce lost sales.

The neural network learns from previous customer behaviour to make predictions.

## **8. Understanding the training process**

The PDF introduces six important ideas on page 6.

| Concept             | Purpose                               |
| ------------------- | ------------------------------------- |
| Weighted sum        | Combines customer information         |
| Sigmoid             | Converts the score into a probability |
| Loss function       | Measures prediction error             |
| Gradient            | Measures how parameters affect error  |
| Gradient descent    | Adjusts parameters to reduce error    |
| Second forward pass | Tests the updated parameters          |

The entire purpose is to make future predictions more accurate.

# **PART 3: UNDERSTANDING THE NEURAL NETWORK MODEL**

## **9. Our training example**

Pages 7–8 introduce the numerical example we will use throughout the lesson.

We have three customer features.

**Customer information**

| Variable  | Meaning                     | Value |
| --------- | --------------------------- | ----- |
| \\(x_1\\) | Time spent on checkout page | 2.0   |
| \\(x_2\\) | Number of items in cart     | 3.0   |
| \\(x_3\\) | Device type (Mobile = 0)    | 0.0   |
| \\(y\\)   | Actual target               | 0.0   |

**Initial parameters**

\\[ w_1=0.5 \\]

\\[ w_2=-0.2 \\]

\\[ w_3=0.4 \\]

\\[ b=0.1 \\]

**Learning rate**

\\[ \alpha=0.1 \\]

Let's understand the values.

- \\(x\\) represents customer information.
- \\(w\\) represents the importance assigned to each feature.
- \\(b\\) is the bias.
- \\(y\\) is the actual answer.
- \\(\alpha\\) controls how much the model changes its parameters during learning.

Our actual target is:

\\[ y=0 \\]

For this example, 0 means the customer did not abandon the cart.

The model must learn to predict this correctly.

## **10. Understanding the architecture**

The PDF uses a single artificial neuron:

[Simple Neural Networks in Python. A detail-oriented introduction to… | by Aidan Wilson | TDS Archive | Medium](https://images.openai.com/static-rsc-4/si90cPp8FefvuX1MrodxCV2OPKfD9YKvL1KPDNe09DvkhsmBxvIreDtA7qlFu_UKTeWVamcY2mdna4S0NrBDzj9gz7azH9PliOPW39I1PjBecYpL8RGTxcgfN7GfVGOpnWifwllxJpvwoLqyivToLr1QSzYuIkseXaUXZqh2Mew?purpose=inline)

[medium.com](https://medium.com/data-science/inroduction-to-neural-networks-in-python-7e0b422e6c24?utm_source=chatgpt.com)

The process is:

\\[ X\rightarrow z\rightarrow \sigma(z)\rightarrow\hat y \\]

Where:

- \\(X\\) = customer inputs.
- \\(z\\) = weighted sum.
- \\(\sigma(z)\\) = sigmoid activation.
- \\(\hat y\\) = predicted probability.

The document explains that this single-neuron model with sigmoid activation is mathematically equivalent to logistic regression.

### **Linear regression versus logistic regression**

**Linear regression**

Predicts continuous numerical values.

For example, predicting someone's salary or a company's sales.

\\[ y=wx+b \\]

**Logistic regression**

Predicts probabilities for classification problems.

For example:

- Customer leaves.
- Customer does not leave.

It uses the sigmoid function to convert the calculated score into a value between 0 and 1.

# **PART 4: FORWARD PROPAGATION**

## **11. Step 1 – Weighted sum**

Forward propagation begins by combining the customer information.

The formula from page 10 is:

\\[ z=\sum\_{i=1}^{n}w_ix_i+b \\]

Don't be frightened by the sigma symbol.

It simply means:

**Multiply each input by its corresponding weight, add the results, then add the bias.**

Our formula becomes:

\\[ z=w_1x_1+w_2x_2+w_3x_3+b \\]

Substitute the values:

\\[ z=(0.5)(2)+(-0.2)(3)+(0.4)(0)+0.1 \\]

Calculate each part:

\\[ z=1.0-0.6+0+0.1 \\]

Therefore:

\\[ \boxed{z=0.5} \\]

**What does 0.5 mean?**

It is the neuron's raw evidence score. It is not yet a probability.

The positive score means the current weighted calculation leans toward predicting abandonment.

### **Why are the weights important?**

Look at the calculation:

| Feature       | Calculation | Contribution |
| ------------- | ----------- | ------------ |
| Checkout time | 0.5 × 2     | 1.0          |
| Cart items    | -0.2 × 3    | -0.6         |
| Device        | 0.4 × 0     | 0            |
| Bias          | 0.1         | 0.1          |
| **Total**     |             | **0.5**      |

Notice that the device contributes zero because its input is zero.

This is how the network combines different customer features.

## **12. Step 2 – Sigmoid activation function**

Now we have:

\\[ z=0.5 \\]

But the company wants a probability.

For example, it wants to know whether the customer has a 20%, 60% or 90% chance of abandoning the cart.

We use the sigmoid function.

The formula is:

\\[ \boxed{\sigma(z)=\frac{1}{1+e^{-z}}} \\]

Let's understand the symbols:

- \\(\sigma\\) = sigmoid function.
- \\(z\\) = weighted sum.
- \\(e\\) = Euler's number, approximately 2.71828.
- The result is always between 0 and 1.

### **Why do we use sigmoid?**

Because probabilities must be between 0 and 1.

For example:

| Sigmoid output | Interpretation  |
| -------------- | --------------- |
| 0.20           | 20% probability |
| 0.50           | 50% probability |
| 0.80           | 80% probability |
| 0.95           | 95% probability |

### **Calculating sigmoid**

Our value is:

\\[ z=0.5 \\]

Substitute it:

\\[ \hat y=\frac{1}{1+e^{-0.5}} \\]

Calculate the exponential:

\\[ e^{-0.5}\approx0.60653 \\]

Add 1:

\\[ 1+0.60653=1.60653 \\]

Divide:

\\[ \hat y=\frac{1}{1.60653} \\]

\\[ \boxed{\hat y=0.6225} \\]

Initial prediction

The model predicts approximately:

\\[ \boxed{62.25\\%} \\]

This means the model estimates a 62.25% probability of cart abandonment.

### **Important: Understanding the sigmoid curve**

[Logistic Regression, the Sigmoid Function, and Decision Boundaries (Made Intuitive) | by Ayush Saxena | Feb, 2026 | Medium](https://images.openai.com/static-rsc-4/bKVbCH1cBXe6ZiYM_ZMfkk3uqs_s2scqPxq8xIxF3SxVJPI0LBMdX3nzHJVhR1U7wc3I4K_UotP42Z7A9_feW1RV4pnZbKZoTgh4uP6ndBIf1kyUdf17j6_bg609nFIafk3ZbtY4GggXZCvyp06729T6Oh_4-X4FyxBzP6JGeyA?purpose=inline)

[medium.com](https://medium.com/%40iamayush027/logistic-regression-the-sigmoid-function-and-decision-boundaries-made-intuitive-2172124bc67f?utm_source=chatgpt.com)

Three important observations:

1. When \\(z\\) is very positive, sigmoid approaches 1.
2. When \\(z\\) is very negative, sigmoid approaches 0.
3. When \\(z=0\\), sigmoid equals 0.5.

### **What is the value of e?**

\\[ e\approx2.71828 \\]

It is a mathematical constant used in exponential functions.

You don't need to memorize a long explanation of Euler's number for this example. You need to know how to use it in the sigmoid calculation.

# **PART 5: MEASURING PREDICTION ERROR**

## **13. Step 3 – Loss function**

Now comes an important question:

How do we know whether our prediction was correct?

Remember:

- Actual answer = 0.
- Predicted answer = 0.6225.

Our model predicted a probability of 62.25%, but the actual target was 0.

We need to measure how far the prediction is from the actual answer.

That is the purpose of a loss function.

### **Mean Squared Error (MSE)**

Your PDF introduces the squared-error calculation.

The formula used in the worked example is:

\\[ \boxed{E=\frac12(y-\hat y)^2} \\]

Where:

- \\(E\\) = error or loss.
- \\(y\\) = actual value.
- \\(\hat y\\) = predicted value.

### **Why do we square the error?**

Suppose the difference is negative:

\\[ 0-0.6225=-0.6225 \\]

If we square it:

\\[ (-0.6225)^2=0.3875 \\]

Squaring removes the negative sign.

It also makes larger errors receive greater penalties.

### **Why multiply by one-half?**

The PDF explains that this makes differentiation easier because the 2 produced by differentiating a squared term cancels with the one-half.

### **Let's calculate the loss**

We know:

\\[ y=0 \\]

\\[ \hat y=0.6225 \\]

Substitute:

\\[ E=\frac12(0-0.6225)^2 \\]

First subtract:

\\[ E=\frac12(-0.6225)^2 \\]

Square:

\\[ E=\frac12(0.3875) \\]

Multiply:

\\[ \boxed{E=0.1937} \\]

**Our prediction error**

Actual value\
**0**

Predicted value\
**0.6225**

Loss

# **0.1937**

The model made an error.

The loss function gives us a numerical measurement of that error.

The smaller the loss, the closer the prediction is to the target under this loss calculation.

*Technical note:* The PDF calls this MSE, but the worked example uses half the squared error for one training example. A dataset-wide MSE would average errors across multiple examples.

# **PART 6: BACKPROPAGATION**

## **14. Step 4 – What is backpropagation?**

Now we know the model made a mistake.

But we need to answer another question:

**Which parameters contributed to the mistake, and how much should they change?**

This is what backpropagation does.

It calculates how the error depends on the weights and bias.

The process uses a mathematical concept called the chain rule.

## **15. Understanding the chain rule**

Your PDF includes a differentiation diagram on page 20.

[Chain Rule/ Implicit/Higher Order Derivatives Quiz 12th Grade Quiz | Wayground](https://images.openai.com/static-rsc-4/erlt8vgNFIULAfnNkSv1LrvNWNY5EkBZDZ7LLok20nrKHzteofOZA4WTpIDEPVKltiq_2ENCGr2KZkgGkk6UuOFXrKOFqQVW4OAVYtCmQMmouNwMhGMpxFnp1loUMlO_cjh3wx2sF903LUxlkqexTJCEdR-MdGxrX9A2crwc1ew?purpose=inline)

[wayground.com](https://wayground.com/admin/quiz/64023327d20048001d306ea0/chain-rule-implicithigher-order-derivatives-quiz?utm_source=chatgpt.com)

Imagine three connected stages:

\\[ w\rightarrow z\rightarrow\hat y\rightarrow E \\]

The weight affects the weighted sum.

The weighted sum affects the prediction.

The prediction affects the error.

Therefore, to calculate how a weight affects the error, we consider all these relationships.

The chain rule formula is:

\\[ \boxed{ \frac{\partial E}{\partial w} = \frac{\partial E}{\partial\hat y} \times \frac{\partial\hat y}{\partial z} \times \frac{\partial z}{\partial w} } \\]

Let's understand each part.

| Derivative                              | Meaning                                     |
| --------------------------------------- | ------------------------------------------- |
| \\(\frac{\partial E}{\partial\hat y}\\) | How prediction changes the error            |
| \\(\frac{\partial\hat y}{\partial z}\\) | How the weighted sum changes the prediction |
| \\(\frac{\partial z}{\partial w}\\)     | How the weight changes the weighted sum     |

Think of it as tracing the effect of a weight through the entire network until it reaches the final error.

## **16. Calculating the error signal**

The PDF uses the following formula:

\\[ \delta=(\hat y-y)\hat y(1-\hat y) \\]

We already know:

\\[ \hat y=0.6225 \\]

\\[ y=0 \\]

First calculate the prediction difference:

\\[ 0.6225-0=0.6225 \\]

Now calculate the sigmoid derivative:

\\[ \hat y(1-\hat y) \\]

\\[ =0.6225(1-0.6225) \\]

\\[ =0.6225(0.3775) \\]

\\[ \approx0.2350 \\]

Multiply:

\\[ \delta=0.6225\times0.2350 \\]

\\[ \boxed{\delta=0.1463} \\]

This is called the local error signal.

It combines the prediction error with the sensitivity of the sigmoid function.

# **PART 7: GRADIENT CALCULATION**

## **17. Step 5 – What is a gradient?**

A gradient measures how much the loss changes when a parameter changes.

In simple English:

It tells us how strongly a weight or bias affects the error.

A large gradient means the parameter has a stronger effect on the loss at that point.

A small gradient means a weaker effect.

The PDF gives us:

\\[ \boxed{\frac{\partial E}{\partial w_i}=\delta x_i} \\]

This means:

Gradient = Error signal × Input value.

Let's calculate the gradients.

### **Gradient for weight 1**

\\[ \frac{\partial E}{\partial w_1} =0.1463\times2 \\]

\\[ \boxed{0.2926} \\]

### **Gradient for weight 2**

\\[ \frac{\partial E}{\partial w_2} =0.1463\times3 \\]

\\[ \boxed{0.4389} \\]

### **Gradient for weight 3**

\\[ \frac{\partial E}{\partial w_3} =0.1463\times0 \\]

\\[ \boxed{0} \\]

### **Gradient for bias**

The bias gradient is:

\\[ \frac{\partial E}{\partial b}=\delta \\]

Therefore:

\\[ \boxed{0.1463} \\]

### **Summary of gradients**

| Parameter | Gradient |
| --------- | -------- |
| \\(w_1\\) | 0.2926   |
| \\(w_2\\) | 0.4389   |
| \\(w_3\\) | 0        |
| \\(b\\)   | 0.1463   |

Notice that weight 2 has the largest gradient.

Weight 3 has a gradient of zero because its corresponding input was zero.

# **PART 8: GRADIENT DESCENT**

## **18. Step 6 – What is gradient descent?**

We now know how each parameter contributes to the error.

The next question is:

How do we improve the model?

We use gradient descent.

Gradient descent is an optimization algorithm that updates the weights and bias to reduce prediction error.

The formula is:

\\[ \boxed{ w\_{new}=w\_{old}-\alpha\frac{\partial E}{\partial w} } \\]

Where:

- \\(w\_{new}\\) = new weight.
- \\(w\_{old}\\) = previous weight.
- \\(\alpha\\) = learning rate.
- \\(\frac{\partial E}{\partial w}\\) = gradient.

### **Why do we subtract the gradient?**

The gradient points in the direction of increasing loss locally.

We want to move in the opposite direction, toward lower loss.

Therefore, we subtract it.

## **19. Understanding the learning rate**

Your PDF uses:

\\[ \alpha=0.1 \\]

The learning rate determines the size of the adjustment.

Think about walking down a hill.

- Very small steps: You move slowly.
- Very large steps: You may overshoot your destination.
- Appropriate steps: You move toward the bottom more effectively.

The PDF gives these illustrations:

| Learning rate | Explanation                          |
| ------------- | ------------------------------------ |
| 0.01          | Very small steps                     |
| 0.1           | Moderate steps                       |
| 0.5           | Larger steps                         |
| 1             | May jump too far and become unstable |

The best learning rate depends on the problem and training process.

## **20. Updating the weights and bias**

Now we apply gradient descent to our original parameters.

Remember:

\\[ \alpha=0.1 \\]

### **Updating weight 1**

Original weight:

\\[ w_1=0.5 \\]

Gradient:

\\[ 0.2926 \\]

Substitute:

\\[ w\_{1,new}=0.5-0.1(0.2926) \\]

\\[ =0.5-0.02926 \\]

\\[ \boxed{w\_{1,new}=0.4707} \\]

### **Updating weight 2**

Original weight:

\\[ w_2=-0.2 \\]

Gradient:

\\[ 0.4389 \\]

Substitute:

\\[ w\_{2,new}=-0.2-0.1(0.4389) \\]

\\[ =-0.2-0.04389 \\]

\\[ \boxed{w\_{2,new}=-0.2439} \\]

### **Updating weight 3**

Original weight:

\\[ w_3=0.4 \\]

Gradient:

\\[ 0 \\]

Therefore:

\\[ w\_{3,new}=0.4-0.1(0) \\]

\\[ \boxed{w\_{3,new}=0.4} \\]

### **Updating bias**

Original bias:

\\[ b=0.1 \\]

Gradient:

\\[ 0.1463 \\]

Substitute:

\\[ b\_{new}=0.1-0.1(0.1463) \\]

\\[ =0.1-0.01463 \\]

\\[ \boxed{b\_{new}=0.0854} \\]

### **Final updated parameters**

**Before and after training**

| Parameter | Old value | New value |
| --------- | --------- | --------- |
| \\(w_1\\) | 0.5000    | 0.4707    |
| \\(w_2\\) | -0.2000   | -0.2439   |
| \\(w_3\\) | 0.4000    | 0.4000    |
| \\(b\\)   | 0.1000    | 0.0854    |

The model has now adjusted its parameters based on the error.

# **PART 9: SECOND FORWARD PASS**

## **21. Step 7 – Making another prediction**

We have updated the weights and bias.

But how do we know whether the changes helped?

We perform another forward pass.

The PDF explains this on page 25.

Our new values are:

\\[ w_1=0.4707 \\]

\\[ w_2=-0.2439 \\]

\\[ w_3=0.4000 \\]

\\[ b=0.0854 \\]

The inputs remain the same:

\\[ x_1=2,\quad x_2=3,\quad x_3=0 \\]

### **Calculate the new weighted sum**

\\[ z\_{new}=w_1x_1+w_2x_2+w_3x_3+b \\]

Substitute:

\\[ z\_{new}=(0.4707)(2)+(-0.2439)(3)+(0.4000)(0)+0.0854 \\]

Calculate:

\\[ =0.9414-0.7317+0+0.0854 \\]

\\[ \boxed{z\_{new}\approx0.2951} \\]

### **Calculate the new sigmoid prediction**

\\[ \hat y\_{new}=\frac{1}{1+e^{-0.2951}} \\]

\\[ \boxed{\hat y\_{new}\approx0.5732} \\]

The new prediction is approximately 57.32%.

**Prediction comparison**

Original prediction\
**62.25%**

Updated prediction\
**57.32%**

Calculated using the updated weights and bias listed in the PDF.

The prediction has moved closer to the actual target of zero.

**Important correction in the PDF:** Page 25 reports a new prediction of 0.6062, but the updated parameters printed on that page produce approximately 0.5732 when substituted into the stated formula. The page 7 infographic also contains inconsistent parameter values. We should use the formula and listed updated parameters when performing the calculation.

# **PART 10: FINAL BUSINESS INTERPRETATION**

## **22. What did the neuron learn?**

Initially, the model predicted:

\\[ 62.25\\% \\]

After updating its parameters, our recalculation gives:

\\[ 57.32\\% \\]

The actual target was:

\\[ 0 \\]

The model reduced its prediction, moving closer to the actual target.

However, the updated prediction is still above 50%.

If we use a classification threshold of 0.5, the model still classifies the shopper as likely to abandon the cart.

This demonstrates something important:

**One training update does not necessarily make a model completely accurate.**

## **23. Why do repeated iterations improve performance?**

The training process repeats several times:

## **ANN training cycle**

1. **Forward propagation**

   Make a prediction.
2. **Loss calculation**

   Measure the prediction error.
3. **Backpropagation**

   Calculate how parameters contributed to the error.
4. **Gradient descent**

   Update weights and bias.
5. **Repeat**

   Make another prediction and continue training.

This process can happen thousands of times.

Over repeated iterations, the network can learn better parameter values and improve its predictions.

The goal is not simply to memorize the training data. It is to learn patterns that also work on new, unseen customer data.

## **24. Connecting mathematics to business**

This is the most important summary of the entire lesson.

| Mathematical concept | Business meaning                             |
| -------------------- | -------------------------------------------- |
| Inputs               | Customer behaviour                           |
| Weights              | Learned importance of behaviours             |
| Bias                 | Baseline adjustment                          |
| Weighted sum         | Combined customer evidence                   |
| Sigmoid              | Probability of abandonment                   |
| Loss                 | Prediction error                             |
| Backpropagation      | Identifying parameter contributions to error |
| Gradient             | Sensitivity of error to parameters           |
| Gradient descent     | Improving model parameters                   |
| Repeated training    | Improving predictive performance             |

The final business objective is to identify customers who may abandon their shopping carts so that the company can intervene earlier and reduce lost sales.

This covers the main concepts, worked calculations and business interpretation from all 27 pages of your PDF.&#x20;

Artificial Neuro networks churn prediction Algorithm overview\.pdf



# **PART 11: EXAM REVISION – IMPORTANT FORMULAS**

Keep these formulas because they are the foundation of this topic.

| Concept            | Formula                                                     |
| ------------------ | ----------------------------------------------------------- |
| Weighted sum       | \\(z=\sum w_ix_i+b\\)                                       |
| Sigmoid            | \\(\sigma(z)=\frac{1}{1+e^{-z}}\\)                          |
| Squared loss       | \\(E=\frac12(y-\hat y)^2\\)                                 |
| Sigmoid derivative | \\(\hat y(1-\hat y)\\)                                      |
| Error signal       | \\(\delta=(\hat y-y)\hat y(1-\hat y)\\)                     |
| Weight gradient    | \\(\frac{\partial E}{\partial w_i}=\delta x_i\\)            |
| Bias gradient      | \\(\frac{\partial E}{\partial b}=\delta\\)                  |
| Gradient descent   | \\(w\_{new}=w\_{old}-\alpha\frac{\partial E}{\partial w}\\) |

## **PART 12: TEST YOUR UNDERSTANDING**

Let's check whether you can now explain the entire topic without relying on memorization.

**ANN revision test**

0/10 answered

**1. What is an ANN?**

A. A system inspired by the human brain that learns patterns

B. A database for storing information

C. A computer operating system



**2. What is the purpose of weights?**

A. To store customer names

B. To determine the influence of inputs

C. To calculate the final loss directly



**3. What does forward propagation produce?**

A. A prediction

B. A gradient

C. A learning rate



**4. Why is sigmoid used in this example?**

A. To produce probabilities between 0 and 1

B. To remove all weights

C. To calculate the number of inputs



**5. What was the initial weighted sum?**

A. 0.1

B. 0.5

C. 0.6225



**6. What was the initial predicted probability?**

A. 50%

B. 80%

C. 62.25%



**7. What does the loss function measure?**

A. Prediction error

B. Number of hidden layers

C. Number of customers



**8. What does backpropagation calculate?**

A. How parameters contribute to prediction error

B. The number of customers

C. The output layer size



**9. What does gradient descent do?**

A. Deletes the training data

B. Updates parameters to reduce loss

C. Stops the network from learning



**10. What is the purpose of the learning rate?**

A. Controls the size of parameter updates

B. Determines customer age

C. Counts the number of predictions



Submit test

**My recommendation:** Complete this test first. Then we can practise the mathematical calculations independently, especially sigmoid, loss, chain rule, gradients and gradient descent. Those are the areas where understanding the steps matters most.
