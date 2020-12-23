// main imports
import React from 'react';
import { split } from "sentence-splitter";
import * as md5 from 'md5';

// utils
import SocketClient from '../utils/socketClient.js'

// Text sentiment visualiser, captures text input, runs sentiment analysis on sentence level, and colors sentences based on sentiment score
class TextSentimentVisualiser extends React.Component {

  // Constructor
  constructor(props) {
    super(props);
    // init state
    this.state = {
      text: '',
      sentences: new Map()
    };

    // init handlers
    this.handleTextAssessmentResponse = this.handleTextAssessmentResponse.bind(this);

    // init color scale
    this.colorScale = {
      min: 0.45,
      max: 0.55
    }
  }

  // Handler for text entry in text area
  handleCapture(capturedText) {
    // split captured text by sentences
    const splitBySentence = split(capturedText);

    // go through all splitted sentences
    const currentSentences = this.state.sentences;
    const updatedSentences = new Map();

    splitBySentence.forEach((element) => {
      if (element['type'] === 'Sentence') {
        // calculate sentence's hash and check if if is already in the sentence's map
        const sentence = element['raw'];
        const key = md5(sentence);

        if (currentSentences.has(key)) {
          // seems like an existing sentence - let's simply copy that to the map
          updatedSentences.set(key, currentSentences.get(key))
        } else {
          // seems like a new sentence found - let's classify add that to the map
          this.classifySentence(key, sentence);
          updatedSentences.set(key, { sentence: sentence, sentiment: -1 });
        }
      }

      // keep updated sentences in state 
      this.setState({ sentences: updatedSentences });
    })

    // keep captured text in state to show that under capture text area
    this.setState({ text: capturedText });
  }

  // Send new sentence for classification
  classifySentence(key, sentence) {
    this.socketClient.sendRequest(JSON.stringify({ key: key, sentence: sentence }));
  }

  // Update state with sentence classification results
  handleTextAssessmentResponse(data) {
    const result = JSON.parse(data);
    const updatedSentences = this.state.sentences;
    updatedSentences.set(result.key, { sentence: result.sentence, sentiment: result.sentiment })
    this.setState({ sentences: updatedSentences });
  }

  // after-mount init
  componentDidMount() {
    this.socketClient = new SocketClient(process.env.REACT_APP_SERVER_URL, this.handleTextAssessmentResponse);
    this.socketClient.openSocket();
  }

  // before un-mount clean-up
  componentWillUmount() {
    this.socketClient.closeSocket();
    this.socketClient = null;
  }

  // render for showing captured and classified text
  renderCapturedText() {
    // prepare an array of sentences
    const sentence_array = new Array();
    this.state.sentences.forEach(element => {
      sentence_array.push(element);
    })

    // prepare a list of phrases, colored by sentiment
    const fullText = sentence_array.map((t) => {
      // update color scale, if neeeded
      if (t.sentiment > 0) {
        this.colorScale.min = (this.colorScale.min > t.sentiment) ? t.sentiment : this.colorScale.min;
        this.colorScale.max = (this.colorScale.max < t.sentiment) ? t.sentiment : this.colorScale.max;
      }

      // get text background color by sentiment score
      const color = this.sentimentToColor(t.sentiment, this.colorScale.min, this.colorScale.max);
      const style = { backgroundColor: color };
      return (
        <p style={style}>{t.sentence}</p>);
    });

    return (
      <div>
        {fullText}
      </div>
    );
  }

  // Converts sentiment to color
  // https://gist.github.com/mlocati/7210513
  sentimentToColor(sentiment, min, max) {
    var base = (max - min);

    if (base == 0) { sentiment = 100; }
    else {
      sentiment = (sentiment - min) / base * 100;
    }
    var r, g, b = 0;
    if (sentiment < 50) {
      r = 255;
      g = Math.round(5.1 * sentiment);
    }
    else {
      g = 255;
      r = Math.round(510 - 5.1 * sentiment);
    }
    var h = r * 0x10000 + g * 0x100 + b * 0x1;
    return '#' + ('000000' + h.toString(16)).slice(-6);
  }

  // render for capture text area
  renderEntry() {
    return (
      <textarea id="capture" name="capture" rows="25" cols="100" onInput={(event) => this.handleCapture(event.target.value)}></textarea>
    );
  }

  // main rendering function
  render() {
    return (
      <div>
        <h3>Enter here sentences to assess sentiment</h3>
        {this.renderEntry()}
        <h3>And here is text sentiment by sentence</h3>
        {this.renderCapturedText()}
      </div>
    );
  }
}

export default TextSentimentVisualiser;