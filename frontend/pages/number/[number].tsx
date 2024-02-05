// pages/[number].tsx
import { useRouter } from 'next/router';

const DoubleNumberPage = () => {
  const router = useRouter();
  const { number } = router.query; // URLからnumberパラメータを取得

  // URLパラメータを数値に変換し、2倍にする
  const result = number ? Number(number) * 2 : "数値がURLに含まれていません";

  return (
    <div>
      <h1>結果: {result}</h1>
    </div>
  );
};

export default DoubleNumberPage;