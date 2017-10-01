import Foundation
import SwiftNeuralNetwork


// Input matrix for 7-segment-display digits 0-9
var input = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],   // 0
             [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],   // 1
             [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],   // 2
             [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],   // 3
             [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],   // 4
             [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],   // 5
             [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],   // 6
             [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],   // 7
             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],   // 8
             [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],   // 9
            ]

// Corresponding output- The correct target digit is represented as a 1, and everything else in set is 0. 
var ideal = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   // 0 
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   // 1
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   // 2
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   // 3
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   // 4
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],   // 5
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],   // 6
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],   // 7
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],   // 8
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],   // 9
             ]

print("Train:")


var neuralNetwork = Network(inputCount: 7, hiddenCount: 4, outputCount: 10, learnRate: 1.0, momentum: 0.9)  // or hiddenCount: 5


for i in 0..<5000 {
	for j in 0..<input.count {
        neuralNetwork.train(input: input[j], ideal: ideal[j])
	}

	print("Epoch # \(i) - Error: \(neuralNetwork.getError(len: input.count))")
}

print("\nTest Accuracy:")

for i in 0..<input.count {
	var s = "\(i) = [ "
	for j in 0..<input[0].count {
		s += "\(input[i][j]), "
	}
	s += "]"

    let output = neuralNetwork.predict(input: input[i])

	s += " => [ "
	for p in 0..<output.count {
		s += "\(String(format:"%.2f", output[p])), "
	}
	s += "]"

    print(s)
}

print("\nTest 9 not 100%")

let output = neuralNetwork.predict(input: [0.8, 1.0, 1.0, 0.0, 0.0, 0.8, 0.7])
for p in 0..<output.count {
    print(String(format:"%.2f", output[p]))
}


