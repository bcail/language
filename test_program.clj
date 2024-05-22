(require [language.string str]
         [language.sqlite3 sqlite]
         ["test_functions.clj" f])

(let [counts {}]
  (loop [line (read-line)]
    (if (not (nil? line))
      (do
        (if (not (str/blank? line))
          (f/process-line counts line))
        (recur (read-line)))))
  (print counts))

(println f/value)

(print (nth (cli-args) 1))
(println (nth (cli-args) 2))

(print (sqlite/version))
