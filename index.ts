/**
 * Obesity Prediction Perceptron
 *
 * This program implements a simple perceptron to predict obesity based on a person's height and weight.
 * It uses a training set to learn the relationship between these factors and obesity, then tests its accuracy
 * on a separate test set. The program allows users to input key parameters for the learning process and
 * make predictions based on custom height and weight inputs.
 *
 * The perceptron uses a binary classification approach, where it predicts whether a person is obese or not
 * based on their BMI (Body Mass Index) calculated from their height and weight.
 */

type BmiStats = {
	mean: number;
	stdDev: number;
};

type Person = {
	height: number; // in centimeters
	weight: number; // in kilograms
};

type PersonWithObesity = {
	person: Person;
	isObese: boolean;
};

type Weights = Array<number>;

// Grouped constants
const BMI_CONSTANTS = {
	MIN_HEIGHT: 120,
	MAX_HEIGHT: 240,
	MIN_WEIGHT: 40,
	MAX_WEIGHT: 280,
	OBESITY_THRESHOLD: 30,
};

const DEFAULTS = {
	TRAINING_SET_SIZE: "1000",
	TEST_SET_SIZE: "100",
	EPOCHS: "100",
	HEIGHT: "170",
	WEIGHT: "70",
};

const LEARNING_RATE_BOUNDS = {
	MIN: 0.001,
	MAX: 0.1,
};

const EXIT_COMMAND = "q";

const COLORS = {
	RED: "\x1b[31m",
	GREEN: "\x1b[32m",
	RESET: "\x1b[0m",
};

const colorize = (text: string, color: string): string =>
	`${color}${text}${COLORS.RESET}`;

const colorPrediction = (text: string, isObese: boolean): string =>
	colorize(text, isObese ? COLORS.RED : COLORS.GREEN);

class Perceptron {
	private weights: Weights;

	constructor(inputSize: number) {
		this.weights = this.initializeWeights(inputSize);
	}

	private initializeWeights(inputSize: number): Weights {
		return Array(inputSize + 1)
			.fill(0)
			.map(() => Math.random() * 2 - 1);
	}

	predict(inputs: Array<number>): number {
		const augmentedInputs = [1, ...inputs];
		const sum = augmentedInputs.reduce(
			(acc, input, index) => acc + input * this.weights[index],
			0,
		);
		return sum > 0 ? 1 : 0;
	}

	updateWeights(
		inputs: Array<number>,
		target: number,
		learningRate: number,
	): void {
		const prediction = this.predict(inputs);
		const error = target - prediction;
		const augmentedInputs = [1, ...inputs];
		this.weights = this.weights.map(
			(weight, index) => weight + learningRate * error * augmentedInputs[index],
		);
	}

	async train(
		trainingSet: Array<PersonWithObesity>,
		epochs: number,
		learningRate: number,
	): Promise<void> {
		const start = performance.now();

		for (let epoch = 0; epoch < epochs; epoch++) {
			for (const { person, isObese } of trainingSet) {
				const normalizedInputs = [
					normalize(
						person.height,
						BMI_CONSTANTS.MIN_HEIGHT,
						BMI_CONSTANTS.MAX_HEIGHT,
					),
					normalize(
						person.weight,
						BMI_CONSTANTS.MIN_WEIGHT,
						BMI_CONSTANTS.MAX_WEIGHT,
					),
				];
				this.updateWeights(normalizedInputs, isObese ? 1 : 0, learningRate);

				// Update progress percentage
				const progress = ((epoch + 1) / epochs) * 100;
				process.stdout.write(`\rProgress: ${progress.toFixed(2)}%`);
			}
		}
		const duration = performance.now() - start;

		console.log(`\nTraining complete in ${duration.toFixed(2)}ms`);

		// Add a delay to show the training completion message
		await new Promise((resolve) => setTimeout(resolve, 1000));
		console.log(); // New line after training is complete
	}

	test(testSet: PersonWithObesity[]): number {
		const correctPredictions = testSet.reduce((acc, { person, isObese }) => {
			const normalizedInputs = [
				normalize(
					person.height,
					BMI_CONSTANTS.MIN_HEIGHT,
					BMI_CONSTANTS.MAX_HEIGHT,
				),
				normalize(
					person.weight,
					BMI_CONSTANTS.MIN_WEIGHT,
					BMI_CONSTANTS.MAX_WEIGHT,
				),
			];
			const prediction = this.predict(normalizedInputs);
			return (
				acc +
				((prediction === 1 && isObese) || (prediction === 0 && !isObese)
					? 1
					: 0)
			);
		}, 0);
		return correctPredictions / testSet.length;
	}
}

const calculateBmiStats = (trainingSet: Array<PersonWithObesity>): BmiStats => {
	const bmiValues = trainingSet.map(({ person }) =>
		calculateBmi(person.height, person.weight),
	);
	const mean = bmiValues.reduce((sum, bmi) => sum + bmi, 0) / bmiValues.length;
	const variance =
		bmiValues.reduce((sum, bmi) => sum + (bmi - mean) ** 2, 0) /
		bmiValues.length;
	return { mean, stdDev: Math.sqrt(variance) };
};

const calculateLearningRate = (
	trainingSetSize: number,
	bmiStats: BmiStats,
): number => {
	const coefficientOfVariation = bmiStats.stdDev / bmiStats.mean;
	const thresholdDistance =
		Math.abs(bmiStats.mean - BMI_CONSTANTS.OBESITY_THRESHOLD) /
		BMI_CONSTANTS.OBESITY_THRESHOLD;
	const learningRate =
		(coefficientOfVariation * thresholdDistance) / Math.sqrt(trainingSetSize);
	return Math.max(
		LEARNING_RATE_BOUNDS.MIN,
		Math.min(LEARNING_RATE_BOUNDS.MAX, learningRate),
	);
};

const randomInRange = (min: number, max: number): number =>
	Math.random() * (max - min) + min;

const randomHeight = (): number =>
	randomInRange(BMI_CONSTANTS.MIN_HEIGHT, BMI_CONSTANTS.MAX_HEIGHT);

const randomWeight = (): number =>
	randomInRange(BMI_CONSTANTS.MIN_WEIGHT, BMI_CONSTANTS.MAX_WEIGHT);

const calculateBmi = (height: number, weight: number): number =>
	weight / (height / 100) ** 2;

const createPerson = (height: number, weight: number): Person => ({
	height,
	weight,
});

const personWithObesity = (person: Person): PersonWithObesity => ({
	person,
	isObese:
		calculateBmi(person.height, person.weight) >=
		BMI_CONSTANTS.OBESITY_THRESHOLD,
});

const generateTrainingSet = (size: number): Array<PersonWithObesity> =>
	Array.from({ length: size }, () =>
		personWithObesity(createPerson(randomHeight(), randomWeight())),
	);

const normalize = (value: number, min: number, max: number): number =>
	(value - min) / (max - min);

const validateInput = (input: string): boolean => /^\d+(\.\d+)?$/.test(input);

const validateRange = (input: string, min: number, max: number): boolean => {
	const value = Number.parseFloat(input);
	return value >= min && value <= max;
};

const validateInputRange = (input: string, min: number, max: number): boolean =>
	validateInput(input) && validateRange(input, min, max);

const makePrediction = (
	perceptron: Perceptron,
	defaultHeight = DEFAULTS.HEIGHT,
	defaultWeight = DEFAULTS.WEIGHT,
): [boolean, string?, string?] => {
	console.log(colorize("\n--------------------", COLORS.GREEN));
	console.log(
		`\nEnter details for prediction (or type ${EXIT_COMMAND} to exit):`,
	);

	const heightInput = prompt(
		`Enter height in cm (default: ${defaultHeight}): `,
	);
	if (heightInput?.toLowerCase() === EXIT_COMMAND) return [false];

	const weightInput = prompt(
		`Enter weight in kg (default: ${defaultWeight}): `,
	);
	if (weightInput?.toLowerCase() === EXIT_COMMAND) return [false];

	const height = Number.parseFloat(heightInput || defaultHeight);
	const weight = Number.parseFloat(weightInput || defaultWeight);

	if (
		!validateInputRange(
			heightInput || defaultHeight,
			BMI_CONSTANTS.MIN_HEIGHT,
			BMI_CONSTANTS.MAX_HEIGHT,
		) ||
		!validateInputRange(
			weightInput || defaultWeight,
			BMI_CONSTANTS.MIN_WEIGHT,
			BMI_CONSTANTS.MAX_WEIGHT,
		)
	) {
		console.log(
			colorize(
				`Invalid input. Please enter values within the accepted range. (W = ${BMI_CONSTANTS.MIN_WEIGHT}-${BMI_CONSTANTS.MAX_WEIGHT}, H = ${BMI_CONSTANTS.MIN_HEIGHT}-${BMI_CONSTANTS.MAX_HEIGHT})`,
				COLORS.RED,
			),
		);
		return [true, defaultHeight, defaultWeight];
	}

	const normalizedInputs = [
		normalize(height, BMI_CONSTANTS.MIN_HEIGHT, BMI_CONSTANTS.MAX_HEIGHT),
		normalize(weight, BMI_CONSTANTS.MIN_WEIGHT, BMI_CONSTANTS.MAX_WEIGHT),
	];

	const prediction = perceptron.predict(normalizedInputs);
	const predictionText = prediction === 1 ? "Obese" : "Not Obese";
	const coloredPrediction = colorPrediction(predictionText, prediction === 1);

	console.log(`Prediction: ${coloredPrediction}`);

	return [true, heightInput || defaultHeight, weightInput || defaultWeight];
};

const getTrainingParameters = (): {
	trainingSetSize: number;
	testSetSize: number;
	epochs: number;
} => {
	console.log(
		"\nTraining set size: The number of samples used to train the perceptron. A larger set may improve accuracy but increase training time.",
	);
	const trainingSetSize = Number.parseInt(
		prompt(
			`Enter the training set size (default: ${DEFAULTS.TRAINING_SET_SIZE}): `,
		) || DEFAULTS.TRAINING_SET_SIZE,
	);

	console.log(
		"\nTest set size: The number of samples used to evaluate the perceptron's accuracy. It should be different from the training set.",
	);
	const testSetSize = Number.parseInt(
		prompt(`Enter the test set size (default: ${DEFAULTS.TEST_SET_SIZE}): `) ||
			DEFAULTS.TEST_SET_SIZE,
	);

	console.log(
		"\nNumber of epochs: The number of times the perceptron will iterate over the entire training set. More epochs may improve accuracy but increase training time.",
	);
	const epochs = Number.parseInt(
		prompt(`Enter the number of epochs (default: ${DEFAULTS.EPOCHS}): `) ||
			DEFAULTS.EPOCHS,
	);

	if (trainingSetSize <= 0 || testSetSize <= 0 || epochs <= 0) {
		throw new Error("Training parameters must be positive numbers");
	}

	return { trainingSetSize, testSetSize, epochs };
};

const trainAndTestPerceptron = async (
	perceptron: Perceptron,
	trainingSet: Array<PersonWithObesity>,
	testSet: Array<PersonWithObesity>,
	epochs: number,
	learningRate: number,
): Promise<number> => {
	console.log("Training perceptron...\n");
	await perceptron.train(trainingSet, epochs, learningRate);

	return perceptron.test(testSet);
};

const runPredictionLoop = (perceptron: Perceptron): void => {
	let currentHeight = DEFAULTS.HEIGHT;
	let currentWeight = DEFAULTS.WEIGHT;

	while (true) {
		const [shouldContinue, height, weight] = makePrediction(
			perceptron,
			currentHeight,
			currentWeight,
		);
		if (!shouldContinue) break;

		currentHeight = height || currentHeight;
		currentWeight = weight || currentWeight;
	}
};

// Main program
const main = async (): Promise<void> => {
	console.log(`
    Welcome to the Obesity Prediction Perceptron!
    
    This program uses a simple neural network (perceptron) to predict obesity based on height and weight.
    The perceptron learns from a training dataset where obesity is determined using Body Mass Index (BMI).
    
    How it works:
    1. We generate a training dataset using BMI calculations.
       (BMI is calculated as weight (kg) / (height (m))^2, with BMI >= 30 considered obese)
    2. The perceptron learns patterns from this dataset, associating height and weight with obesity.
    3. After training, the perceptron can make predictions based on new height and weight inputs.
    
    Note: While based on BMI data, the perceptron's predictions may not exactly match BMI calculations.
    This demonstration is for educational purposes and should not be used for medical diagnosis.
    Always consult with a healthcare professional for health-related concerns.
    `);

	const { trainingSetSize, testSetSize, epochs } = getTrainingParameters();
	const trainingSet = generateTrainingSet(trainingSetSize);
	const testSet = generateTrainingSet(testSetSize);
	const bmiStats = calculateBmiStats(trainingSet);
	const learningRate = calculateLearningRate(trainingSetSize, bmiStats);

	const perceptron = new Perceptron(2);

	const accuracy = await trainAndTestPerceptron(
		perceptron,
		trainingSet,
		testSet,
		epochs,
		learningRate,
	);

	console.log(colorize("\n\nTraining Summary", COLORS.GREEN));

	console.log(`
    Training Set Size: ${trainingSetSize}
    BMI Mean: ${bmiStats.mean.toFixed(2)}
    BMI StdDev: ${bmiStats.stdDev.toFixed(2)}
    Test Set Size: ${testSetSize}
    Epochs: ${epochs}
    Learning Rate: ${learningRate.toFixed(6)}
    Accuracy: ${(accuracy * 100).toFixed(2)}%
    `);

	console.log(`
While the perceptron achieved an accuracy of ${(accuracy * 100).toFixed(2)}% on the test set,
its accuracy is limited and predictions may not match BMI calculations.
  `);

	console.log(
		colorize(
			"\nPerceptron training complete. You can now make predictions.",
			COLORS.GREEN,
		),
	);
	runPredictionLoop(perceptron);

	console.log("\nBye!");
};

// Run the program
main();
