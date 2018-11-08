import abc


class TestTemplate(abc.ABC):

    @classmethod
    def setUpClass(cls):
        cls.sos_token = '<sos>'
        cls.eos_token = '</s>'
        cls.pad_token = '<pad>'
        cls.unk_token = '<unk>'
        cls.maxlen = 7
        cls.input_data = [
            '克安是牛肉大粉絲',  # longer than 7 after adding sos eos
            '繼良喜歡喝星巴巴',  # longer than 7 after adding sos eos
            '安靜的祥睿',  # equal to 7 after adding sos eos
            '喔',  # shorter than 7 after adding sos eos
        ]

    def setUp(self):
        self.indexer = self.get_indexer()
        self.indexer.fit(self.input_data)

    @abc.abstractmethod
    def get_indexer(self):
        pass

    @abc.abstractmethod
    def get_correct_idxs_and_seqlen_of_input_data(self):
        pass

    def test_word2index_out_of_range(self):
        with self.assertRaises(KeyError):
            self.indexer.word2index('凢')

    def test_index2word_out_of_range(self):
        with self.assertRaises(KeyError):
            self.indexer.index2word(100000000000)

    def test_transform(self):
        tx_data, meta = self.indexer.transform(self.input_data)
        correct_idxs, correct_seqs = self.get_correct_idxs_and_seqlen_of_input_data()
        self.assertEqual(
            correct_idxs,
            tx_data,
        )
        self.assertEqual(
            correct_seqs,
            meta['seqlen'],
        )

    def test_inverse_transform(self):
        tx_data, meta = self.indexer.transform(self.input_data)
        output = self.indexer.inverse_transform(tx_data, meta['inv_info'])
        self.assertEqual(
            output,
            self.input_data,
        )
