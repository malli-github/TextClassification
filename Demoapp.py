{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\Chindukuri\\.conda\\envs\\tensorflow\\lib\\site-packages\\tensorflow\\python\\ops\\resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Colocations handled automatically by placer.\n",
      "Loaded model from disk\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import sys\n",
    "import numpy as np\n",
    "import keras\n",
    "from keras.preprocessing.text import Tokenizer\n",
    "from keras.utils import to_categorical\n",
    "from keras.preprocessing.sequence import pad_sequences\n",
    "from keras.models import model_from_json\n",
    "from keras.models import load_model\n",
    "from keras.models import Model\n",
    "json_file = open('model.json', 'r')\n",
    "loaded_model_json = json_file.read()\n",
    "json_file.close()\n",
    "loaded_model = model_from_json(loaded_model_json)\n",
    "loaded_model.load_weights(\"best_model.h5\")\n",
    "print(\"Loaded model from disk\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "   WARNING: This is a development server. Do not use it in a production deployment.\n",
      "   Use a production WSGI server instead.\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)\n",
      "127.0.0.1 - - [13/Nov/2020 15:55:37] \"\u001b[37mGET / HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [13/Nov/2020 15:55:37] \"\u001b[33mGET /favicon.ico HTTP/1.1\u001b[0m\" 404 -\n",
      "127.0.0.1 - - [13/Nov/2020 15:55:46] \"\u001b[37mPOST / HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [13/Nov/2020 15:55:59] \"\u001b[37mGET / HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [13/Nov/2020 15:56:04] \"\u001b[37mPOST / HTTP/1.1\u001b[0m\" 200 -\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, render_template,request\n",
    "app = Flask(__name__,template_folder='./templates2')\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('input.html')\n",
    "@app.route('/',methods=['POST'])\n",
    "def test_data():\n",
    "    Test_texts=[]\n",
    "    texts=[]\n",
    "    fpath=request.form['fname']\n",
    "    f = open(fpath, encoding='latin-1')\n",
    "    t = f.read()\n",
    "    texts.append(t)\n",
    "    MAX_WORDS = 10000\n",
    "    MAX_SEQUENCE_LENGTH = 1000\n",
    "    tokenizer  = Tokenizer(num_words = MAX_WORDS)\n",
    "    tokenizer.fit_on_texts(texts)\n",
    "    sequences =  tokenizer.texts_to_sequences(texts)\n",
    "    Test_word_index = tokenizer.word_index\n",
    "    #print(\"unique words : {}\".format(len(Test_word_index)))\n",
    "    Test_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)\n",
    "    #return Test_data\n",
    "    test_lables_set={'alt.atheism': 0, 'comp.graphics': 1, 'comp.os.ms-windows.misc': 2, 'comp.sys.ibm.pc.hardware': 3, 'comp.sys.mac.hardware': 4, 'comp.windows.x': 5, 'misc.forsale': 6, 'rec.autos': 7, 'rec.motorcycles': 8, 'rec.sport.baseball': 9, 'rec.sport.hockey': 10, 'sci.crypt': 11, 'sci.electronics': 12, 'sci.med': 13, 'sci.space': 14, 'soc.religion.christian': 15, 'talk.politics.guns': 16, 'talk.politics.mideast': 17, \n",
    "     'talk.politics.misc': 18, 'talk.religion.misc': 19}\n",
    "    labels1=list(test_lables_set.keys())\n",
    "    Test_labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]\n",
    "    Test_labels = to_categorical(np.asarray(Test_labels))\n",
    "    label=[]\n",
    "    for x_t in Test_data:\n",
    "        prediction = loaded_model.predict(np.array([x_t]))\n",
    "        predicted_label = labels1[np.argmax(prediction[0])]\n",
    "    #prediction = loaded_model.predict(np.array([Test_data]))\n",
    "    #predicted_label=labels1[np.argmax(prediction[0])]\n",
    "        label.append(predicted_label)\n",
    "    return render_template('result.html',fp=str(predicted_label))\n",
    "#print(label)\n",
    "if __name__ == '__main__':\n",
    "    app.run(threaded=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
